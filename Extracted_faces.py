import numpy as np
import face_recognition as fr


class ExtractedFace:
    def __init__(self, id_frame, frame, face_location, prob=None):
        self.id_last_frame = id_frame
        self.last_frame = frame
        
        self.last_face_location = face_location
        self.last_face_image = self.__cut_face()    # for testing # WARNING IS POINTER from frame not copy
        self.last_face_encoding = self.__encode_image()
        
        self.last_face_probability = prob
    
    def __encode_image(self):
        face_encoding = fr.face_encodings(self.last_frame, [self.last_face_location])
        if len(face_encoding) > 1: raise Exception("found more than face in location")
        return face_encoding

    def __cut_face(self):
        t, r, b, l = self.last_face_location
        return self.last_frame[l:r,t:b]
        
    def match_by_encodings(self, known_faces_encodings:list, tolerance=0.6) -> list:
        matches = fr.compare_faces(known_faces_encodings, self.last_face_encoding, tolerance=tolerance)
        return matches
    
    def distance_by_encodings(self, known_faces_encodings:list) -> list:
        res = fr.face_distance(known_faces_encodings, self.last_face_encoding)
        return res

class TrackedFace:
    
    ID = 0

    def __init__(self, face:ExtractedFace):
        self.ID = TrackedFace.ID

        self.__dict__.update(vars(face))
        
        self.best_face_image = self.last_face_image
        self.best_face_probability = self.last_face_probability

        self.duration_existence = 1
        TrackedFace.ID += 1
    
    def update_info(self, face:ExtractedFace):
        """call when sucess track"""

        # self.last_frame = face.last_frame
        # self.last_face_location = face.last_face_location
        # self.last_face_image = face.last_face_image
        # self.last_face_encoding = face.last_face_encoding
        self.__dict__.update(vars(face))
        
        if self.best_face_probability <= self.last_face_probability:
            self.best_face_image = self.last_face_image
            self.best_face_probability = self.last_face_probability
        
        self.duration_existence += 1
        pass


    # def similarity(first_obj:"TrackedFace", second_obj:"TrackedFace"):
    #     return np.dot(first_obj.features_encoding, second_obj.features_encoding) / (
    #         np.linalg.norm(first_obj.features_encoding)*np.linalg.norm(second_obj.features_encoding)
    #         )
    