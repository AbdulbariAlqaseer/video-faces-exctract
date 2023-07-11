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

# Loop through each frame in the video using VideoCapture.read()
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

    face_locations = fr.face_locations(frame[:,:,::-1])
    # face_encodings = fr.face_encodings(frame[:,:,::-1], face_locations)

    print(face_locations)

    # sleep(0.04)
    # if cv2.waitKey(1) == ord('q'):
    #   break

# save data as a dataframe
# save_data(SAVE_DF_PATH, data, columns=["id_frame", "start_point", "end_point", "name_category", "probability"])

vid.release()
cv2.destroyAllWindows() 

