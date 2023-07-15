import os
from utile import index_to_time, save_data, save_image, visualize, detection_info
from config import SAVE_DF_PATH, MODEL_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import numpy as np
import pandas as pd
from time import sleep
import face_recognition as fr
import cv2
from face import ExtractedFace, TrackedFace
from os.path import join
from detector_model import Detector, FaceReconationDetecor, MediaPipeDetector
from typing import Union

def get_info_result(detection_result, frame):
    faces_image = locations = prob = []
    for detection in detection_result.detections:
        bbox = detection.bounding_box   
        top, left = bbox.origin_x, bbox.origin_y
        bottom, right = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        faces_image.append(
            frame[left:right, top:bottom].copy()
        )
        locations.append(
            (top, right, bottom, left)
        )
        prob.append(detection.categories[0].score)
    return faces_image, locations, prob


def detect_faces(id_frame, frame):
    face_locations = fr.face_locations(frame)
    return [ExtractedFace(id_frame, frame, face_location_set) for face_location_set in face_locations]


def save_all_faces(all_faces:list[TrackedFace], fps):
    if not os.path.exists(SAVE_IMAGE_PATH):
        os.mkdir(SAVE_IMAGE_PATH)
    # data as csv
    print(f"length data: {len(all_faces)}")
    data = [face_set.to_csv() for face_set in all_faces]
    df = pd.DataFrame(data, columns=data[0].keys())
    
    df["start_time"] =  df.start_index_frame.apply(index_to_time, fps=fps)
    df["end_time"] =  df.end_index_frame.apply(index_to_time, fps=fps)

    df.to_csv(SAVE_DF_PATH, index=False)

    # save best faces
    image_paths = [join(SAVE_IMAGE_PATH, face_set.get_unique_name() + ".jpg")  for face_set in all_faces] 
    images = [face_set.best_face_image for face_set in all_faces]
    for path, image in zip(image_paths, images):
        save_image(image[:,:,::-1], path)








# Create an FaceDetector object.
# VisionRunningMode = mp.tasks.vision.RunningMode
# base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO)
# detector = vision.FaceDetector.create_from_options(options)

all_faces, tracked_faces, current_faces = [], [], []

# Use OpenCV’s VideoCapture to load the input video.
vid = cv2.VideoCapture(VIDEO_PATH)

# Load the frame rate of the video
fps = vid.get(cv2.CAP_PROP_FPS)
print(f"{fps = }")

detector = FaceReconationDetecor()

while(vid.isOpened()):
      
    # Capture the video frame
    frame_exists, frame = vid.read()
    
    # if frame is read correctly ret is True
    # if not frame_exists: break

    # get fram position and frame millesecond time
    num_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
    ms = vid.get(cv2.CAP_PROP_POS_MSEC)

    # to take frame in second
    if num_frame % int(fps) : continue
    # resize
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    # convert frame to RGB
    frame = frame[:,:,::-1]

    # detect faces
    current_faces = detector.detect_faces(num_frame, frame)

    # cv2.imshow("frame", frame[:,:,::-1])
    # for i, face in enumerate(current_faces):
    #     cv2.imshow(f'face image {i}', face.last_face_image)

    Faces_keep_appearing = []
    if tracked_faces:
        current_faces_encodings = [face.last_face_encoding for face in current_faces]

        # Compare between all last faces and all current_faces
        for index, tracked_face_set in enumerate(tracked_faces):
            tracked_face_set:TrackedFace

            if len(current_faces_encodings) == 0: 
                all_faces.append(tracked_face_set)
                continue

            matches = tracked_face_set.match_by_encodings(current_faces_encodings)
            face_distances = tracked_face_set.distance_by_encodings(current_faces_encodings)

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                tracked_face_set.update_info(current_faces[best_match_index])
                current_faces.pop(best_match_index)
                current_faces_encodings.pop(best_match_index)
                Faces_keep_appearing.append(tracked_face_set)
            else:
                all_faces.append(tracked_face_set)
    
    tracked_faces = Faces_keep_appearing
    for face in current_faces:
        tracked_faces.append(TrackedFace(face))


    if cv2.waitKey(1) == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
save_all_faces(all_faces, fps)


class FaceDetectionTimeTracker:
    def __init__(self, video_path, model: Union[Detector, str], n_sec: int = 1, resize_to:tuple[int, int] = None):
        if (not isinstance(model, Detector)) or (not model in ["face-reconation", "mediapipe"]): 
            raise Exception('model must be "face-reconation" or "mediapipe", or object from "Detector"')
        
        self.vid = cv2.VideoCapture(video_path)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)

        if model == "face-reconation":
            self.detector = FaceReconationDetecor()
        elif model == "mediapipe":
            self.detector = MediaPipeDetector()
        else:
            self.detector = model

        self.all_faces, self.tracked_faces, self.current_faces = [], [], []
        self.n_sec = n_sec
        self.dsize = resize_to
    
    def run(self):
        while(True):
            # Capture the video frame
            frame_exists, frame = vid.read()
            
            # if frame is read correctly ret is True
            if not frame_exists: break

            # get fram position and frame millesecond time
            num_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

            # to take frame in second
            if num_frame % int(self.n_sec * fps) : continue
            
            # resize
            if self.dsize: frame = cv2.resize(frame, self.dsize)

            # detect faces
            current_faces = self.detector.detect_faces(num_frame, frame)

            Faces_keep_appearing = []
            if self.tracked_faces:
                current_faces_encodings = [face.last_face_encoding for face in self.current_faces]

                # Compare between all last faces and all current_faces
                for index, tracked_face_set in enumerate(self.tracked_faces):
                    tracked_face_set:TrackedFace

                    if len(current_faces_encodings) == 0: 
                        all_faces.append(tracked_face_set)
                        continue

                    matches = tracked_face_set.match_by_encodings(current_faces_encodings)
                    face_distances = tracked_face_set.distance_by_encodings(current_faces_encodings)

                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        tracked_face_set.update_info(self.current_faces[best_match_index])
                        current_faces.pop(best_match_index)
                        current_faces_encodings.pop(best_match_index)
                        Faces_keep_appearing.append(tracked_face_set)
                    else:
                        self.all_faces.append(tracked_face_set)
            
            self.tracked_faces = Faces_keep_appearing
            for face in self.current_faces:
                self.tracked_faces.append(TrackedFace(face))