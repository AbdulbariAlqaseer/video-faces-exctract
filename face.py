import numpy as np
import face_recognition as fr
import cv2 as cv
from utile import index_to_time
# from deepface import DeepFace

# class ExtractedFace:
#     def __init__(self, frame, face_location, id_frame=None, face_box=None, prob=None):
#         self.id_last_frame = id_frame
#         self.last_frame = frame
#         self.face_box = face_box
#         self.last_face_location = face_location
#         self.last_face_image = self.__cut_face()    # for testing # WARNING IS POINTER from frame not copy
#         self.last_face_encoding = self.__encode_image()
        
#         self.last_face_probability = prob
    
#     def __encode_image(self):
#         face_encoding = fr.face_encodings(self.last_frame, [self.last_face_location])
#         if len(face_encoding) > 1: raise Exception("found more than face in location")
#         return face_encoding[0]

#     def __cut_face(self):
#         t, r, b, l = self.last_face_location
#         return self.last_frame[t:b, l:r]
        

class TrackedFace:
    
    ID = 0

    def __init__(self, frame, face_location, id_frame=None, prob=None):
        self.face_ID = TrackedFace.ID

        self.id_last_frame = self.id_first_frame = id_frame
        self.duration_existence = 0

        self.last_frame = frame
        self.last_face_location = face_location
        self.last_face_image = self.__cut_face()    # for testing # WARNING IS POINTER from frame not copy
        self.last_face_encoding = self.__encode_image()
        self.last_face_probability = prob

        self.best_face_image = self.last_face_image
        self.best_face_probability = self.last_face_probability
        self.best_face_encoding = self.last_face_encoding

        TrackedFace.ID += 1
    
    def __encode_image(self):
        face_encoding = fr.face_encodings(self.last_frame, [self.last_face_location])
        if len(face_encoding) > 1: raise Exception("found more than face in location")
        return face_encoding[0]

    def __cut_face(self):
        t, r, b, l = self.last_face_location
        return self.last_frame[t:b, l:r]

    def update_info(self, face:"TrackedFace"):
        """call when sucess track"""
        assert isinstance(face, TrackedFace)

        """
            self.id_last_frame          Y
            self.id_first_frame         N
            self.duration_existence     calc
            
            self.last_frame             Y
            self.last_face_location     Y
            self.last_face_image        Y
            self.last_face_encoding     Y
            self.last_face_probability  Y

            self.best_face_image        calc
            self.best_face_probability  calc
            self.best_face_encoding     calc
        """
        self.id_last_frame = face.id_last_frame
        self.last_frame = face.last_frame
        self.last_face_location = face.last_face_location
        self.last_face_image = face.last_face_image
        self.last_face_encoding = face.last_face_encoding
        self.last_face_probability = face.last_face_probability
        
        if self.best_face_probability and self.best_face_probability <= face.last_face_probability:
            self.best_face_image = self.last_face_image
            self.best_face_encoding = self.last_face_encoding
            self.best_face_probability = self.last_face_probability
        
        self.duration_existence = self.id_last_frame - self.id_first_frame 

    def match_by_encodings(self, known_faces_encodings:list, tolerance=0.6) -> list:
        matches = fr.compare_faces(known_faces_encodings, self.last_face_encoding, tolerance=tolerance)
        return matches
    
    def distance_by_encodings(self, known_faces_encodings:list) -> list:
        res = fr.face_distance(known_faces_encodings, self.last_face_encoding)
        return res
    
    def get_unique_name(self):
        return f"ID{self.face_ID}-fromFrame{self.id_first_frame}_toFrame{self.id_last_frame}"

    def to_dict(self):
        return {
            "start_index_frame" : self.id_first_frame,
            "end_index_frame" : self.id_last_frame,
            "duration_existence" : self.duration_existence,
            "encoding" : self.best_face_encoding,
            "face_image" : self.best_face_image 
        }

'''
class FaceTrack:
    
    ID = 0

    def __init__(self,frame, face:ExtractedFace ):
        self.face_ID = FaceTrack.ID

        assert isinstance(face, ExtractedFace)
        self.__dict__.update(vars(face))
        
        self.id_first_frame = self.id_last_frame
        tracker = cv.TrackerKCF_create()
        # tracker  = cv.legacy.TrackerMOSSE_create()
        ret = tracker.init(frame, self.face_box)
        self.tracker = tracker
        self.tracked_face_image = self.get_tracked_face()

        self.best_face_image = self.last_face_image
        self.best_face_probability = self.last_face_probability
        self.best_face_encoding = self.last_face_encoding
        self.duration_existence = 0
        FaceTrack.ID += 1

    def update_info(self , id_frame , frame):
        """call when sucess track"""
        success , bbox = self.tracker.update(frame)
        if success:
            self.id_last_frame = id_frame
            self.last_frame = frame
            self.face_box = bbox
            self.tracked_face_image = self.get_tracked_face()
            self.duration_existence += 1
        

    def get_tracked_face(self ):
        return self.last_frame[self.face_box[1]:self.face_box[1]+ self.face_box[3] ,self.face_box[0]:self.face_box[0]+ self.face_box[2]]
    
    def check_detected2track(self ,id_frame ,face:ExtractedFace ):
            model_name = 'Facenet'
            result = DeepFace.verify(self.tracked_face_image, face.last_face_image, model_name=model_name , enforce_detection= False )
            if result['verified']:
                self.id_last_frame = id_frame
                if face.last_face_probability > self.last_face_probability:
                    self.face_box = face.face_box
                    self.last_face_location = face.last_face_location
                    self.last_face_image = face.last_face_image
                    self.last_face_probability = face.last_face_probability

                self.duration_existence += 1
                return True
            else:
                return False

    def add_to_track(self ,face:ExtractedFace):
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

'''