import numpy as np
import face_recognition.api
class extracted_faces:
    
    THRESHOLD = 0.2
    ID = 0

    def __init__(self, frame_id, face_image, prob):
        self.ID = extracted_faces.ID
        self.frame_id = frame_id
        self.last_face_image = face_image    # for testing
        self.last_face_encoding = self.__encode_image()
        self.best_face_image = face_image
        self.probability = prob
        self.duration_existence = 1

        extracted_faces.ID += 1

    def __encode_image(self):
        return None

    def increase_duration(self):
        self.duration_existence += 1

    def set_best_face_image(self, face_image, prob):
        if self.probability <= prob:
            self.best_face_image = face_image

    def similarity(first_obj:"extracted_faces", second_obj:"extracted_faces"):
        return np.dot(first_obj.features_encoding, second_obj.features_encoding) / (
            np.linalg.norm(first_obj.features_encoding)*np.linalg.norm(second_obj.features_encoding)
            )
    
    def __eq__(self, another_obj:"extracted_faces"):
        return extracted_faces.similarity(self, another_obj) >= extracted_faces.THRESHOLD