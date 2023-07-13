"""
    detect with mediapipe, comapre with face_reconation, work with opencv
"""
from utile import save_data, visualize, detection_info
from config import SAVE_DF_PATH, MODEL_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import numpy as np
from time import sleep
import face_recognition as fr
import cv2
from face import ExtractedFace, TrackedFace


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
    print(face_locations)
    return [ExtractedFace(id_frame, frame, face_location_set) for face_location_set in face_locations]




# Create an FaceDetector object.
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO)
detector = vision.FaceDetector.create_from_options(options)

all_faces, tracked_faces, current_faces = [], [], []

# Use OpenCV’s VideoCapture to load the input video.
vid = cv2.VideoCapture(VIDEO_PATH)

# Load the frame rate of the video
fps = int(vid.get(cv2.CAP_PROP_FPS))
print(f"{fps = }")

while(True):
      
    # Capture the video frame
    frame_exists, frame = vid.read()
    
    # if frame is read correctly ret is True
    if not frame_exists: break

    # get fram position and frame millesecond time
    num_frame = vid.get(cv2.CAP_PROP_POS_FRAMES)
    ms = vid.get(cv2.CAP_PROP_POS_MSEC)

    # to take frame in second
    if num_frame % fps : continue

    # resize
    frame = cv2.resize(frame, (512, 512))
    
    # convert frame to RGB
    frame = frame[:,:,::-1]
    
    # detect faces
    current_faces = detect_faces(num_frame, frame)

    cv2.imshow("frame", frame[:,:,::-1])
    for i, face in enumerate(current_faces):
        cv2.imshow(f'face image {i}', face.last_face_image)

    # if last_clip_faces:
    #     # Compare between all last faces and all current_faces
    #     pass


    if cv2.waitKey(1) == ord('q'):
        break

