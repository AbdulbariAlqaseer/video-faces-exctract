from utile import index_to_time, save_data, visualize, detection_info
from config import SAVE_DF_PATH, MODEL_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import numpy as np
from time import sleep
import cv2

# Create an FaceDetector object.
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO)
detector = vision.FaceDetector.create_from_options(options)

data = []

# Use OpenCV’s VideoCapture to load the input video.
vid = cv2.VideoCapture(VIDEO_PATH)

# Load the frame rate of the video
fps = vid.get(cv2.CAP_PROP_FPS)
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
    if num_frame % int(fps) : continue
    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Detect faces in the input image.
    # The face detector must be created with the video mode.
    face_detector_result = detector.detect_for_video(mp_image, int(ms))

    # Process the detection result. In this case, visualize it.
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, face_detector_result)
    
    # cv2.imwrite(f"{SAVE_IMAGE_PATH}\\{num_frame}.png", annotated_image)
    cv2.imshow("vid after detect", annotated_image)
    try:
        (start_point, end_point, category_name, probability, result_text, text_location) = next(detection_info(face_detector_result))
        cv2.imshow("only first face", annotated_image[start_point[1]:end_point[1], start_point[0]:end_point[0]])
    except:
       pass
    # collect info for every face to save it
    # for (start_point, end_point, category_name, probability, result_text, text_location) in\
    #     detection_info(face_detector_result):
    #     data.append([num_frame, start_point, end_point, category_name, probability])

    sleep(0.04)
    if cv2.waitKey(1) == ord('q'):
      break

# save data as a dataframe
# save_data(SAVE_DF_PATH, data, columns=["id_frame", "start_point", "end_point", "name_category", "probability"])

vid.release()
cv2.destroyAllWindows()

