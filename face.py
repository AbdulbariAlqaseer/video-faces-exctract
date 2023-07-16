import numpy as np
import face_recognition as fr

from utile import index_to_time

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
        return face_encoding[0]

    def __cut_face(self):
        t, r, b, l = self.last_face_location
        return self.last_frame[t:b, l:r]
        

class TrackedFace:
    
    ID = 0

    def __init__(self, face:ExtractedFace):
        self.face_ID = TrackedFace.ID

        assert isinstance(face, ExtractedFace)
        self.__dict__.update(vars(face))
        
        self.id_first_frame = self.id_last_frame
        
        self.best_face_image = self.last_face_image
        self.best_face_probability = self.last_face_probability
        self.best_face_encoding = self.last_face_encoding

        self.duration_existence = 1
        TrackedFace.ID += 1
    
    def update_info(self, face:ExtractedFace):
        """call when sucess track"""
        assert isinstance(face, ExtractedFace)
        self.__dict__.update(vars(face))
        
        if self.best_face_probability and self.best_face_probability <= self.last_face_probability:
            self.best_face_image = self.last_face_image
            self.best_face_probability = self.last_face_probability
            self.best_face_encoding = self.last_face_encoding
        
        self.duration_existence += 1
        pass

    def match_by_encodings(self, known_faces_encodings:list, tolerance=0.6) -> list:
        matches = fr.compare_faces(known_faces_encodings, self.last_face_encoding, tolerance=tolerance)
        return matches
    
    def distance_by_encodings(self, known_faces_encodings:list) -> list:
        res = fr.face_distance(known_faces_encodings, self.last_face_encoding)
        return res
    
    def get_unique_name(self):
        return f"ID{self.face_ID}-fromFrame{self.id_first_frame}_toFrame{self.id_last_frame}"

    def to_csv(self):
        d = {
            "start_index_frame" : self.id_first_frame,
            "end_index_frame" : self.id_last_frame,
            "duration_frames" : self.duration_existence,
            "unique_name" : self.get_unique_name()
        }
        return d

    