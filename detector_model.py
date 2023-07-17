from abc import ABC
from face import ExtractedFace
import face_recognition
from functools import partial
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp

class Detector(ABC):
    def __init__(self) -> None:
        pass

    def detect_faces(self, id_frame:int, frame:np.ndarray, **kwargs) -> list[ExtractedFace]:
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
                        detection.bounding_box.origin_x
                    ) for detection in detection_result.detections
                ]

class FaceReconationDetecor(Detector):
    def __init__(self, number_of_times_to_upsample: int = 1, model: str = 'hog') -> None:
        super().__init__()
        self.__model_detect_locations = partial(face_recognition.face_locations, number_of_times_to_upsample = number_of_times_to_upsample, model = model)

    def detect_faces(self, id_frame:int, frame:np.ndarray, **kwargs) -> list[ExtractedFace]:
        """Detects faces in the frame

        Args:
            id_frame (int): id of frame, frame order number
            frame (np.ndarray): opencv frame in mode BGR

        Returns:
            list[ExtractedFace]: Return a list of extracted faces that each expresses a face and its information
        """
        face_locations = self.__model_detect_locations(frame[:,:,::-1])
        return [ExtractedFace(id_frame, frame, face_location_set) for face_location_set in face_locations]