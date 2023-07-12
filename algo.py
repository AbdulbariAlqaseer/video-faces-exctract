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


def detect_result(detection_result, frame):
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



# Create an FaceDetector object.
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO)
detector = vision.FaceDetector.create_from_options(options)

data = all_faces = last_clip_faces = []

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

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect faces in the input image.
    # The face detector must be created with the video mode.
    face_detector_result = detector.detect_for_video(mp_image, int(ms))

    # crop all faces
    current_faces_images, current_faces_locations, current_faces_prob = detect_result(face_detector_result, frame)
    current_faces_encodings = fr.face_encodings(frame, current_faces_locations)

    if last_clip_faces:
        # Compare between all last faces and all current_faces
        
        


        pass


