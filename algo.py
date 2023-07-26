import os
from utile import index_to_time, save_data, save_image, visualize, detection_info
from config import SAVE_DF_PATH, MODEL_MEDIAPIPE_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
# from mediapipe.tasks.python import vision
# from mediapipe.tasks import python
# import mediapipe as mp
import numpy as np
import pandas as pd
from time import sleep
import face_recognition
import cv2
import fastface as ff
from face import ExtractedFace, TrackedFace , FaceTrack
from os.path import join
from detector_model import Detector, FaceReconationDetecor, MediaPipeDetector , FastFaceDetector
from typing import Union

class FaceDetectionTimeTracker:
    def __init__(self, video_path, model: Union[Detector, str] = "face-reconation", n_sec: int = 1, resize_to: tuple[int, int] = None, memorize_face_sec: int = 0, model_mediapipe_path = MODEL_MEDIAPIPE_PATH):
        if (not isinstance(model, Detector)) and (not model in ["face-reconation", "mediapipe","fast-face"]): 
            raise Exception('model must be "face-reconation" or "mediapipe or fast-face", or object from "Detector"')
        
        self.vid = cv2.VideoCapture(video_path)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

        if model == "face-reconation":
            self.detector = FaceReconationDetecor()
        elif model == "mediapipe":
            self.detector = MediaPipeDetector(model_mediapipe_path)
        
        elif model == "fast-face":
            self.detector = FastFaceDetector()
        else:
            self.detector = model

        self.window_size_frames = int(n_sec * self.fps)
        self.dsize = resize_to
        self.memorize_face_frames = int(memorize_face_sec * self.fps)
    
    def run(self, save_image_path, save_df_path):
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
            # self.current_faces = [ExtractedFace(num_frame, frame, face_location_set) for face_location_set in face_locations]
            self.current_faces = self.detector.detect_faces(num_frame, frame, ms=ms)
            
            Faces_keep_appearing = []
            if self.tracked_faces:
                current_faces_encodings = [face.last_face_encoding for face in self.current_faces]

                # Compare between all last faces and all current_faces
                for tracked_face_set in self.tracked_faces:
                    tracked_face_set:TrackedFace

                    if num_frame - tracked_face_set.id_last_frame - self.window_size_frames > self.memorize_face_frames:
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
                self.tracked_faces.append(TrackedFace(face))
        
        self.all_faces.extend(self.tracked_faces)
        # if len(self.all_faces) > 0:
        self.save_data(save_image_path, save_df_path)
    

    def run2(self , save_image_path , save_df_path , det_thre):
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
            self.save_data(save_image_path, save_df_path)



    def run3(self , save_image_path , save_df_path , det_thre):
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
            tracked_faces_encodings = [track.best_face_encoding for track in self.tracked_faces]
            # Compare between all tracked faces and current_detected_faces
            for face in self.current_faces:
                if len(tracked_faces_encodings):
                    matches = face.match_by_encodings(tracked_faces_encodings)
                    if any(matches):
                        face_distances = face.distance_by_encodings(tracked_faces_encodings)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            self.tracked_faces[best_match_index].id_last_frame = num_frame
                            if face.last_face_probability > self.tracked_faces[best_match_index].last_face_probability:
                                self.tracked_faces[best_match_index].face_box = face.face_box
                                self.tracked_faces[best_match_index].last_face_location = face.last_face_location
                                self.tracked_faces[best_match_index].last_face_image = face.last_face_image
                                self.tracked_faces[best_match_index].last_face_probability = face.last_face_probability

                            self.tracked_faces[best_match_index].duration_existence += 1
                            break
                    else:
                        self.tracked_faces.append(FaceTrack(frame ,face))
        
        self.all_faces.extend(self.tracked_faces)
        if len(self.all_faces) > 0:
            self.save_data(save_image_path, save_df_path)


    def save_data(self, save_image_path, save_df_path):
        if not os.path.exists(save_image_path):
            os.mkdir(save_image_path)
        # data as csv
        print(f"length data: {len(self.all_faces)}")
        data = [face_set.to_csv() for face_set in self.all_faces]
        df = pd.DataFrame(data, columns=data[0].keys())
        
        df["start_time"] =  df.start_index_frame.apply(index_to_time, fps=self.fps)
        df["end_time"] =  df.end_index_frame.apply(index_to_time, fps=self.fps)

        df.to_csv(save_df_path, index=False)

        # save best faces
        image_paths = [join(save_image_path, face_set.get_unique_name() + ".jpg")  for face_set in self.all_faces] 
        images = [face_set.last_face_image for face_set in self.all_faces]
        for path, image in zip(image_paths, images):
            save_image(image, path)



algo = FaceDetectionTimeTracker(VIDEO_PATH, model="fast-face", memorize_face_sec=15)
algo.run2(SAVE_IMAGE_PATH, SAVE_DF_PATH , 0.80 )