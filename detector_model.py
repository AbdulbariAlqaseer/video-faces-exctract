from abc import ABC
from face import ExtractedFace
import face_recognition
from functools import partial

class Detector(ABC):
    def __init__(self) -> None:
        pass

    def detect_faces(self, frame) -> list[ExtractedFace]:
        pass

class MediaPipeDetector(Detector):
    def __init__(self) -> None:
        super().__init__()
        pass

class FaceReconationDetecor(Detector):
    def __init__(self, number_of_times_to_upsample: int = 1, model: str = 'hog') -> None:
        super().__init__()
        self.__model_detect_locations = partial(face_recognition.face_locations, number_of_times_to_upsample = number_of_times_to_upsample, model = model)

    def detect_faces(self, id_frame, frame) -> list[ExtractedFace]:
        face_locations = self.__model_detect_locations(frame)
        return [ExtractedFace(id_frame, frame, face_location_set) for face_location_set in face_locations]