from abc import ABC
from face import TrackedFace
import face_recognition
from functools import partial
import numpy as np
import fastface as ff
import cv2 as cv
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
from typing import Union

class Detector(ABC):
    def __init__(self) -> None:
        pass

    def detect_faces(self,  image:np.ndarray, id_:Union[int, None] = None, **kwargs) -> list[TrackedFace]:
        """Detects faces in the image

        Args:
            image (np.ndarray): opencv image in mode BGR
            id_ (int): id of frame if in video.
        Returns:
            list[TrackedFace]: Return a list of extracted faces that each expresses a face and its information
        """
        pass

class MediaPipeDetector(Detector):
    def __init__(self, model_path) -> None:
        super().__init__()
        # Create an FaceDetector object.
        VisionRunningMode = mp.tasks.vision.RunningMode
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO)
        self.__detector = vision.FaceDetector.create_from_options(options)
        self.__mp_Image = partial(mp.Image, image_format = mp.ImageFormat.SRGB)
    
    def detect_faces(self, image:np.ndarray, id_:Union[int, None] = None, **kwargs) -> list[TrackedFace]:
        mp_image = self.__mp_Image(data=image)
        
        if id_:
            face_detector_result = self.__detector.detect_for_video(mp_image, int(kwargs['ms']))
        else:
            face_detector_result = self.__detector.detect(mp_image) 

        face_locations = self.__convert_results(face_detector_result)
        return [TrackedFace(image, face_location_set, id_frame=id_) for face_location_set in face_locations]


    def __convert_results(self, detection_result):
        return  [
                    (
                        detection.bounding_box.origin_y,
                        detection.bounding_box.origin_x + detection.bounding_box.height,
                        detection.bounding_box.origin_y + detection.bounding_box.width,
                        detection.bounding_box.origin_x,
                        
                    ) for detection in detection_result.detections
                ]

class FaceReconationDetecor(Detector):

    def __init__(self, number_of_timescurrent_faces_to_upsample: int = 1, model: str = 'hog') -> None:
        super().__init__()
        self.model_name = model
        # self.__model_detect_locations = partial(face_recognition.face_locations, number_of_times_to_upsample = number_of_timescurrent_faces_to_upsample, model = model)

    def detect_faces(self, image:np.ndarray, id_:Union[int, None] = None, **kwargs) -> list[TrackedFace]:
        face_locations = face_recognition.face_locations(image[:,:,-1], model=self.model_name)
        current_faces = [TrackedFace(image, face_location_set, id_frame=id_) for face_location_set in face_locations]
        return current_faces
    

class FastFaceDetector(Detector):
    def __init__(self) -> None:
        super().__init__() 
        detector = ff.FaceDetector.from_pretrained("lffd_original")
        detector.eval()
        self.__detector = detector

    def detect_faces(self, image:np.ndarray, id_:Union[int, None] = None, **kwargs) -> list[TrackedFace]:
        # image = image[:,:,::-1]
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        face_detector_result = self.__detector.predict(image, det_threshold=kwargs['det_threshold'])
        face_detector_result = face_detector_result[0]
        corrected_result = self.__check_results(face_detector_result)
        face_locations = self.__convert_results(corrected_result['boxes'])
        face_boxes = self.__create_face_box(corrected_result['boxes'])
        face_probabilities = corrected_result['scores']
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        return [TrackedFace(image, face_location_set, id_frame=id_ ,face_box=face_boxes_set , prob=face_probabilities_set ) for face_location_set , face_probabilities_set , face_boxes_set in zip(face_locations , face_probabilities , face_boxes)]
    
    def __convert_results(self , detection_result):
        return  [
            (
                detection[1],
                detection[2],
                detection[3],
                detection[0],
                
            ) for detection in detection_result
        ]
    
    def __check_results(self , detection_result):
        boxes = np.array(detection_result['boxes'])
        boxes = np.maximum(boxes , 0)
        detection_result['boxes'] = boxes.tolist()
        return detection_result
    
    def __create_face_box(self, detection_result):
        return [
            (
                detection[0],
                detection[1],
                detection[2] - detection[0],
                detection[3] - detection[1]

            ) for detection in detection_result

        ]
