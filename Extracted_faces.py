import numpy as np
import face_recognition as fr

class Face:

    def __init__(self, frame_id, frame, face_location, prob=None):
        self.frame_id = frame_id
        self.last_frame = frame
        
        self.last_face_location = face_location
        self.last_face_image = self.__cut_face()    # for testing # WARNING IS POINTER from frame not copy
        self.last_face_encoding = self.__encode_image()
        
        self.probability = prob
    
    def __encode_image(self):
        face_encoding = fr.face_encodings(self.last_frame, [self.last_face_location])
        if len(face_encoding) > 1: raise Exception("found more than face in location")
        return face_encoding

    def __cut_face(self):
        t, r, b, l = self.last_face_location
        return self.last_frame[l:r,t:b]


class ExtractedFace(Face):
    def __init(self, frame_id, frame, face_location, prob=None):
        pass
        

class TrackedFace:
    
    THRESHOLD = 0.2
    ID = 0

    def __init__(self, frame_id, frame, face_image, face_location, prob):
        self.ID = TrackedFace.ID
        self.duration_existence = 1

        TrackedFace.ID += 1

    def increase_duration(self):
        self.duration_existence += 1

    def set_last_face_image(self, face_image):
        self.last_face_image = face_image

    def set_best_face_image(self, face_image):
        self.best_face_image = face_image

    def similarity(first_obj:"TrackedFace", second_obj:"TrackedFace"):
        return np.dot(first_obj.features_encoding, second_obj.features_encoding) / (
            np.linalg.norm(first_obj.features_encoding)*np.linalg.norm(second_obj.features_encoding)
            )
    
    def __eq__(self, another_obj:"TrackedFace"):
        return TrackedFace.similarity(self, another_obj) >= TrackedFace.THRESHOLD
    
    def update_info(self, obj:"TrackedFace"):
        # self.set_last_frame(obj.last_frame)
        # self.set_last_face_image(obj.last_face_image)
        # self.set_last_face_location(obj.last_face_location)
        # self.set_last_face_encoding(obj.last_face_encoding)
        pass