import os
from utile import index_to_time, save_data, save_image, visualize, detection_info
from config import SAVE_DF_PATH, MODEL_MEDIAPIPE_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
import numpy as np
import pandas as pd
from time import sleep
import cv2
# import fastface as ff
from face import ExtractedFace, TrackedFace , FaceTrack
from os.path import join
from detector_model import Detector, FaceReconationDetecor, MediaPipeDetector
from typing import Union

class FaceDetectionTimeTracker:
    def __init__(self, video_path, model: Union[Detector, str] = "face-reconation", n_sec: int = 1, resize_to: tuple[int, int] = None, memorize_face_sec: int = 0, model_mediapipe_path = MODEL_MEDIAPIPE_PATH ):
        if (not isinstance(model, Detector)) and (not model in ["face-reconation", "mediapipe","fast-face"]): 
            raise Exception('model must be "face-reconation" or "mediapipe or fast-face", or object from "Detector"')
        
        self.vid = cv2.VideoCapture(video_path)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        if model == "face-reconation":
            self.detector = FaceReconationDetecor()
        elif model == "mediapipe":
            self.detector = MediaPipeDetector(model_mediapipe_path)
        
        # elif model == "fast-face":
        #     self.detector = FastFaceDetector()
        else:
            self.detector = model
        
        self.n_sec = n_sec
        self.window_size_frames = int(n_sec * self.fps)
        self.dsize = resize_to
        self.memorize_face_frames = int(memorize_face_sec * self.fps)
    
    def run(self , detect_thre = 0.80):
        self.all_faces, self.tracked_faces, self.current_faces = [], [], []
        while(True):
            # Capture the video frame
            frame_exists, frame = self.vid.read()
            
            # if frame is read correctly ret is True
            if not frame_exists: break

            # get fram position and frame millesecond time
            num_frame = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            ms = self.vid.get(cv2.CAP_PROP_POS_MSEC)

            # to take frame in second
            if num_frame % self.window_size_frames : continue
            
            # resize
            if self.dsize: frame = cv2.resize(frame, self.dsize)

            # detect faces
            # face_locations = face_recognition.face_locations(frame[:,:,-1])
            # self.current_faces = [TrackedFace(num_frame, frame, face_location_set) for face_location_set in face_locations]
            self.current_faces = self.detector.detect_faces(image=frame, id_=num_frame, ms=ms , det_threshold = detect_thre)
            
            Faces_keep_appearing = []
            if self.tracked_faces:
                current_faces_encodings = [face.last_face_encoding for face in self.current_faces]

                # Compare between all last faces and all current_faces
                for tracked_face_set in self.tracked_faces:
                    tracked_face_set:TrackedFace

                    if num_frame - tracked_face_set.id_last_frame > self.memorize_face_frames:
                        self.all_faces.append(tracked_face_set)
                        continue

                    if len(current_faces_encodings):
                        matches = tracked_face_set.match_by_encodings(current_faces_encodings)
                        face_distances = tracked_face_set.distance_by_encodings(current_faces_encodings)

                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            tracked_face_set.update_info(self.current_faces[best_match_index])
                            self.current_faces.pop(best_match_index)
                            current_faces_encodings.pop(best_match_index)

                    Faces_keep_appearing.append(tracked_face_set)
            
            self.tracked_faces = Faces_keep_appearing
            for face in self.current_faces:
                self.tracked_faces.append(face)
        
        self.all_faces.extend(self.tracked_faces)
        if len(self.all_faces) > 0:
            return self.get_result()
        return None
    

    def run2(self ,det_thre=0.80):
        self.all_faces, self.tracked_faces, self.current_faces = [], [], []
        while(True):
            # Capture the video frame
            frame_exists, frame = self.vid.read()
            
            # if frame is read correctly ret is True
            if not frame_exists: break

            # get fram position and frame millesecond time
            num_frame = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            
            # resize
            if self.dsize: frame = cv2.resize(frame, self.dsize)

            # update tracking
            for tracked_face_set in self.tracked_faces:
                tracked_face_set: FaceTrack
                tracked_face_set.update_info(num_frame , frame)

            # keep tracked_faces
            self.all_faces.extend([track for track in self.tracked_faces if num_frame - track.id_last_frame - self.window_size_frames > self.memorize_face_frames])
            self.tracked_faces = [track for track in self.tracked_faces if num_frame - track.id_last_frame - self.window_size_frames <= self.memorize_face_frames]

            # to take frame in second
            if num_frame % self.window_size_frames : continue

            # detect faces
            self.current_faces = self.detector.detect_faces(num_frame, frame, det_threshold = det_thre)
            
            for face in self.current_faces:
                face_found = False
                # Compare between all tracked faces and current_detected_faces
                for tracked_face_set in self.tracked_faces:
                    tracked_face_set:FaceTrack

                    if tracked_face_set.check_detected2track(num_frame ,face):
                        face_found = True
                        break
                if not face_found:
                    self.tracked_faces.append(FaceTrack(frame ,face))
        
        self.all_faces.extend(self.tracked_faces)
        if len(self.all_faces) > 0:
            return self.get_result()
        return None


    def run3(self ,det_thre=0.80):
        self.all_faces, self.tracked_faces, self.current_faces = [], [], []
        while(True):
            # Capture the video frame
            frame_exists, frame = self.vid.read()
            
            # if frame is read correctly ret is True
            if not frame_exists: break

            # get fram position and frame millesecond time
            num_frame = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            
            # resize
            if self.dsize: frame = cv2.resize(frame, self.dsize)

            # update tracking
            for tracked_face_set in self.tracked_faces:
                tracked_face_set: FaceTrack
                tracked_face_set.update_info(num_frame , frame)

            # keep tracked_faces
            self.all_faces.extend([track for track in self.tracked_faces if num_frame - track.id_last_frame - self.window_size_frames > self.memorize_face_frames])
            self.tracked_faces = [track for track in self.tracked_faces if num_frame - track.id_last_frame - self.window_size_frames <= self.memorize_face_frames]
            # to take frame in second
            if num_frame % self.window_size_frames : continue

            # detect faces
            self.current_faces = self.detector.detect_faces(frame, num_frame, det_threshold = det_thre)
            current_faces_encodings = [face.last_face_encoding for face in self.current_faces]
            # tracked_faces_encodings = [track.best_face_encoding for track in self.tracked_faces]
            
            # Compare between all tracked faces and current_detected_faces
            for tracked_face_set in self.tracked_faces:
                tracked_face_set:FaceTrack

                if len(current_faces_encodings):
                    matches = tracked_face_set.match_by_encodings(current_faces_encodings)
                    face_distances = tracked_face_set.distance_by_encodings(current_faces_encodings)

                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        tracked_face_set.add_to_track(self.current_faces[best_match_index])
                        self.current_faces.pop(best_match_index)
                        current_faces_encodings.pop(best_match_index)

            for face in self.current_faces:
                self.tracked_faces.append(FaceTrack(frame ,face))


        
        self.all_faces.extend(self.tracked_faces)
        if len(self.all_faces) > 0: 
            return self.get_result()
        return None
        
    def get_result(self):
        # data as csv

        for face_set in self.all_faces:
            cv2.imwrite(SAVE_IMAGE_PATH +'/'+ face_set.get_unique_name()+'.jpg', face_set.last_face_image)
        data = [ face_set.to_dict() for face_set in self.all_faces]
        data = [ result for result in data if (index_to_time(result['end_index_frame'], fps=self.fps) - index_to_time(result['start_index_frame'],fps=self.fps)) > 5  ]

        df = pd.DataFrame(data, columns=data[0].keys())
        df["start_time"] = df.start_index_frame.apply(index_to_time, fps=self.fps)
        df["end_time"] = df.end_index_frame.apply(index_to_time, fps=self.fps)

        df["duration_existance"] = df["duration_existance"].apply(lambda x:x*self.n_sec)
        
        # images = [face_set.best_face_image for face_set in self.all_faces]
        # encodings = [face_set.best_face_encoding for face_set in self.all_faces]

        return df



# algo = FaceDetectionTimeTracker(VIDEO_PATH, model="face-reconation", memorize_face_sec=15)
# df = algo.run()
# df.to_csv(SAVE_DF_PATH , index = False)