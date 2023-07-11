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

