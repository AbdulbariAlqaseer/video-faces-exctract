import os
from os.path import join, basename, splitext

MARGIN = 2  # pixels
ROW_SIZE = 1  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

MODEL_PATH = 'D:\\python-project\\mediapipe\\detector.tflite'
VIDEO_PATH = 'D:\\final-project\\video-test\\excited-german-fans-celebrating-their-teams-victory.mp4'
NAME_VIDEO = splitext(basename(VIDEO_PATH))[0]  # exrtact name of video without suffix  


SAVE_PATH = "D:\\final-project\\face-reco\\images_result"
SAVE_IMAGE_PATH = join(SAVE_PATH, NAME_VIDEO)
SAVE_DF_PATH = join(SAVE_PATH, NAME_VIDEO, "extracted_faces.csv")