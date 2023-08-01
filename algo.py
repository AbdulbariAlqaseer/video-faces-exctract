from utile import index_to_time
from config import SAVE_DF_PATH, MODEL_MEDIAPIPE_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
import numpy as np
import pandas as pd
from time import sleep
import cv2
from face import ExtractedFace, TrackedFace , FaceTrack
from os.path import join
from detector_model import Detector, FaceReconationDetecor, MediaPipeDetector, FastFaceDetector
from typing import Union

class FaceDetectionTimeTracker:
    def __init__(self, model: Union[Detector, str] = "face-reconation", model_mediapipe_path = MODEL_MEDIAPIPE_PATH):
        if (not isinstance(model, Detector)) and (not model in ["face-reconation", "mediapipe","fast-face"]): 
            raise Exception('model must be "face-reconation" or "mediapipe or fast-face", or object from "Detector"')
        
        if model == "face-reconation":
            self.detector = FaceReconationDetecor()
        elif model == "mediapipe":
            self.detector = MediaPipeDetector(model_mediapipe_path)
        elif model == "fast-face":
            self.detector = FastFaceDetector()
        else:
            self.detector = model 
    
    def from_video(
                self, video_path, resize_to: tuple[int, int] = None,
                n_sec: int = 1, memorize_face_sec: int = 0, 
                detect_threshold = 0.80, existance_threshold = 5
            ):
        res = self.__extract_from_video_depend_by_classic_algo(
                video_path, resize_to, 
                n_sec, memorize_face_sec, 
                detect_threshold, existance_threshold
            )
        return res
        



    def __extract_from_video_depend_by_classic_algo(
                self, video_path, dsize, 
                n_sec, memorize_face_sec, 
                detect_threshold, existance_threshold
            ):
        vid = cv2.VideoCapture(video_path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        window_size_frames = int(n_sec * fps)
        memorize_face_frames = int(memorize_face_sec * fps)

        detect_threshold = detect_threshold
        existance_threshold = existance_threshold
 

        all_faces, tracked_faces, current_faces = [], [], []

        while(True):
            # Capture the video frame
            frame_exists, frame = vid.read()
            # if frame is read correctly ret is True
            if not frame_exists: break

            # get fram position and frame millesecond time
            num_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
            ms = vid.get(cv2.CAP_PROP_POS_MSEC)
            
            # to take frame in second
            if num_frame % window_size_frames : continue
            
            # resize
            if dsize: frame = cv2.resize(frame, dsize)

            # detect faces
            # face_locations = face_recognition.face_locations(frame[:,:,-1])
            # current_faces = [ExtractedFace(num_frame, frame, face_location_set) for face_location_set in face_locations]
            current_faces = self.detector.detect_faces(num_frame, frame, ms=ms , det_threshold = detect_threshold)
            
            Faces_keep_appearing = []
            if tracked_faces:
                current_faces_encodings = [face.last_face_encoding for face in current_faces]

                # Compare between all last faces and all current_faces
                for tracked_face_set in tracked_faces:
                    tracked_face_set:TrackedFace

                    if num_frame - tracked_face_set.id_last_frame > memorize_face_frames:
                        all_faces.append(tracked_face_set)
                        continue

                    if len(current_faces_encodings):
                        matches = tracked_face_set.match_by_encodings(current_faces_encodings)
                        face_distances = tracked_face_set.distance_by_encodings(current_faces_encodings)

                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            tracked_face_set.update_info(current_faces[best_match_index])
                            current_faces.pop(best_match_index)
                            current_faces_encodings.pop(best_match_index)

                    Faces_keep_appearing.append(tracked_face_set)
            
            tracked_faces = Faces_keep_appearing
            for face in current_faces:
                tracked_faces.append(TrackedFace(face))
        
        all_faces.extend(tracked_faces)
        print(len(all_faces))
        if len(all_faces) > 0:
            return self.__get_result_video(all_faces, fps, existance_threshold)
        return None
    
    def __get_result_video(self, all_faces:list[ExtractedFace], fps, existance_threshold):
        return [face_set.to_dict() for face_set in all_faces]
