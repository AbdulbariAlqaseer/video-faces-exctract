from abc import ABC
from face import ExtractedFace
import face_recognition
from functools import partial
import numpy as np
# import fastface as ff
import cv2 as cv
# from mediapipe.tasks.python import vision
# from mediapipe.tasks import python
# import mediapipe as mp

class Detector(ABC):
    def __init__(self) -> None:
        pass

    def detect_faces(self, id_frame:int, frame:np.ndarray, **kwargs) -> list[ExtractedFace]:
        pass

class MediaPipeDetector(Detector):
    def __init__(self, model_path) -> None:
        super().__init__()
        # Create an FaceDetector object.
        # VisionRunningMode = mp.tasks.vision.RunningMode
        # base_options = python.BaseOptions(model_asset_path=model_path)
        # options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO)
        # self.__detector = vision.FaceDetector.create_from_options(options)
        # self.__mp_Image = partial(mp.Image, image_format = mp.ImageFormat.SRGB)
    
    def detect_faces(self, id_frame:int, frame:np.ndarray, **kwargs) -> list[ExtractedFace]:
        mp_frame = self.__mp_Image(data=frame)
        face_detector_result = self.__detector.detect_for_video(mp_frame, int(kwargs['ms']))
        face_locations = self.__convert_results(face_detector_result)
        return [ExtractedFace(id_frame, frame, face_location_set) for face_location_set in face_locations]


    def __convert_results(self, detection_result):
        # locations = []
        # for detection in detection_result.detections:
        #     bbox = detection.bounding_box   
        #     top, left = bbox.origin_x, bbox.origin_y
        #     bottom, right = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        #     locations.append(
        #         (top, right, bottom, left)
        #     )
        # return locations
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

    def detect_faces(self, id_frame:int, frame:np.ndarray, **kwargs) -> list[ExtractedFace]:
        """Detects faces in the frame

        Args:
            id_frame (int): id of frame, frame order number
            frame (np.ndarray): opencv frame in mode BGR

        Returns:
            list[ExtractedFace]: Return a list of extracted faces that each expresses a face and its information
        """
        face_locations = face_recognition.face_locations(frame[:,:,-1], model=self.model_name)
        current_faces = [ExtractedFace(id_frame, frame, face_location_set) for face_location_set in face_locations]
        return current_faces
    
'''
class FastFaceDetector(Detector):
    def __init__(self) -> None:
        super().__init__() 
        detector = ff.FaceDetector.from_pretrained("lffd_original")
        detector.eval()
        self.__detector = detector

    def detect_faces(self, id_frame: int, frame: np.ndarray, **kwargs) -> list[ExtractedFace]:
        # frame = frame[:,:,::-1]
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        face_detector_result = self.__detector.predict(frame, det_threshold=kwargs['det_threshold'])
        face_detector_result = face_detector_result[0]
        corrected_result = self.__check_results(face_detector_result)
        face_locations = self.__convert_results(corrected_result['boxes'])
        face_boxes = self.__create_face_box(corrected_result['boxes'])
        face_probabilities = corrected_result['scores']
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        return [ExtractedFace(id_frame, frame, face_location_set ,face_boxes_set , face_probabilities_set ) for face_location_set , face_probabilities_set , face_boxes_set in zip(face_locations , face_probabilities , face_boxes)]
    
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
'''     
         



        

