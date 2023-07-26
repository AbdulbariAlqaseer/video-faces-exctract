from utile import save_data, visualize, detection_info
from config import SAVE_DF_PATH, MODEL_MEDIAPIPE_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import numpy as np
from time import sleep
import face_recognition as fr
import cv2


# Create an FaceDetector object.
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path=MODEL_MEDIAPIPE_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO)
detector = vision.FaceDetector.create_from_options(options)


def mediapipe_detection(frame):
    # print("mediapipe module : ", end="\t")
    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Detect faces in the input image.
    # The face detector must be created with the video mode.
    face_detector_result = detector.detect_for_video(mp_image, int(ms))
    
    # Process the detection result. In this case, visualize it.
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, face_detector_result)

    def __convert_results(detection_result):
        # locations = []
        # for detection in detection_result.detections:
        #     bbox = detection.bounding_box   
        #     top, left = bbox.origin_x, bbox.origin_y
        #     bottom, right = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        #     locations.append(
        #         (top, right, bottom, left)
        #     )
        # return locations
        return  [
                    (
                        detection.bounding_box.origin_y,
                        detection.bounding_box.origin_x + detection.bounding_box.height,
                        detection.bounding_box.origin_y + detection.bounding_box.width,
                        detection.bounding_box.origin_x
                    ) for detection in detection_result.detections
                ]
    
    face_locations = __convert_results(face_detector_result)
    if len(face_locations):
        t, r, b, l = face_locations[0]
        cv2.imshow("first face mediapipe", annotated_image[t:b, l:r])

    # print("\n")
    return annotated_image

def facerecognition_detection(frame):
    # print("face-reconition : ", end="\t")
    # Convert the frame received from OpenCV and detect
    face_locations = fr.face_locations(frame[:,:,::-1])
    tmp = frame.copy()
    # Process the detection result. In this case, visualize it.
    for t, r, b, l in face_locations:
        # print(f"top = {t}, right = {r}, bottom = {b}, left = {l}", end="\t")
        cv2.rectangle(tmp, (l, t), (r, b), (255, 0, 0), 1)
    # print("\n")
    if len(face_locations):
        t, r, b, l = face_locations[0]
        cv2.imshow("first face face-recognition", tmp[t:b, l:r])
    return tmp


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
    if num_frame % (fps) : continue

    dsize = (
        frame.shape[1] * 0.6,
        frame.shape[0] * 0.6
    )
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    annotated_image = mediapipe_detection(frame)
    cv2.imshow("media pipe", annotated_image)

    annotated_image = facerecognition_detection(frame)
    cv2.imshow("facerecognition_detection", annotated_image)
    
    
    sleep(0.05)
    if cv2.waitKey(1) == ord('q'):
      break

vid.release()
cv2.destroyAllWindows()
