from utile import save_data, visualize, detection_info
from config import SAVE_DF_PATH, MODEL_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import numpy as np
from time import sleep
import face_recognition as fr
import cv2

def detect_result(detection_result, frame, extention=0):
    
    faces_image, locations, prob = [], [], []

    for detection in detection_result.detections:
        bbox = detection.bounding_box   
        top, left = bbox.origin_x, bbox.origin_y
        bottom, right = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        faces_image.append(
            frame[left-extention : right + extention, top-extention:bottom+extention].copy()
        )
        locations.append(
            (top, right, bottom, left)
        )
        prob.append(detection.categories[0].score)
    return faces_image, locations, prob

# Create an FaceDetector object.
VisionRunningMode = mp.tasks.vision.RunningMode
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.IMAGE)
detector = vision.FaceDetector.create_from_options(options)

Ronaldo = cv2.imread("D:\\final-project\\face-reco\\images_test\\cristiano.jfif")
Ronaldo = cv2.resize(Ronaldo, (512, 512))
mp_Ronaldo = mp.Image(image_format=mp.ImageFormat.SRGB, data=Ronaldo)
face_detector_result = detector.detect(mp_Ronaldo)
Ronaldo_face_image, Ronaldo_face_location, Ronaldo_prob_detect = detect_result(face_detector_result, Ronaldo)
Ronaldo_face_encoding = fr.face_encodings(Ronaldo[:,:,::-1])[0]
# cv2.imshow("Ronaldo", Ronaldo)
# cv2.imshow("Ronaldo-face", Ronaldo_face_image[0])

messi = cv2.imread("D:\\final-project\\face-reco\\images_test\\messi.jfif")
messi = cv2.resize(messi, (512, 512))
mp_messi = mp.Image(image_format=mp.ImageFormat.SRGB, data=messi)
face_detector_result = detector.detect(mp_messi)
messi_face_image, messi_face_location, messi_prob_detect = detect_result(face_detector_result, messi)
messi_face_encoding = fr.face_encodings(messi[:,:,::-1])[0]
# cv2.imshow("Messi", messi)
# cv2.imshow("Messi-face", messi_face_image[0])

known_faces_encodings = [Ronaldo_face_encoding, messi_face_encoding]
known_faces_names = ['Ronaldo', 'Messi']


unknown = cv2.imread("D:\\final-project\\face-reco\\images_test\\cistiano2.jfif")
unknown = cv2.resize(unknown, (512, 512))
mp_unknown = mp.Image(image_format=mp.ImageFormat.SRGB, data=unknown)
face_detector_result = detector.detect(mp_unknown)
unknown_faces_images, locations_unknown, probs_unknown = detect_result(face_detector_result, unknown)
encodings_unknown = fr.face_encodings(unknown[:,:,::-1])
cv2.imshow("UNKNOWN", unknown)
cv2.imshow("unknown-face", unknown_faces_images[0])
# t, r, b, l = locations_unknown[0]
# cv2.rectangle(unknown, (t, l), (b,r), (255, 0, 0), 1)

tmp = unknown.copy()
for (t, r, b, l), encoding_unknown in zip(locations_unknown, encodings_unknown):
    matches = fr.compare_faces(known_faces_encodings, encoding_unknown)
    print(matches)
    # Instead, use face_distance to calculate similarities
    face_distances = fr.face_distance(known_faces_encodings, encoding_unknown)
    best_match_index = np.argmin(face_distances)
    name = "Unknown"
    if matches[best_match_index]:
        name = known_faces_names[best_match_index]
    # Draw a box around the face using the Pillow module
    tmp = cv2.rectangle(tmp, (t, l), (b, r), (255, 0, 0), 1)
    # Draw a label with a name below the face
    cv2.putText(tmp, name, (t, l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

cv2.imshow("after recongnition", tmp)
for i, unknown_face_image in enumerate(unknown_faces_images):
    cv2.imshow(f"face {i}", unknown_face_image)
cv2.waitKey(0)