import numpy as np

class extracted_faces:
    
    THRESHOLD = 0.2

    def __init__(self, boundary_box, frame_id):
        self.boundary_box = boundary_box
        self.frame_id = frame_id
        self.duration_existence = 1
        self.features_encoding = None
    
    def increase_duration(self):
        self.duration_existence += 1

    def similarity(first_obj:"extracted_faces", second_obj:"extracted_faces"):
        return np.dot(first_obj.features_encoding, second_obj.features_encoding) / (
            np.linalg.norm(first_obj.features_encoding)*np.linalg.norm(second_obj.features_encoding)
            )
    
    def __eq__(self, another_obj:"extracted_faces"):
        return extracted_faces.similarity(self, another_obj) >= extracted_faces.THRESHOLD