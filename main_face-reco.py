from utile import save_data, visualize, detection_info
from config import SAVE_DF_PATH, MODEL_MEDIAPIPE_PATH, SAVE_IMAGE_PATH, VIDEO_PATH
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
base_options = python.BaseOptions(model_asset_path=MODEL_MEDIAPIPE_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, running_mode=VisionRunningMode.IMAGE)
detector = vision.FaceDetector.create_from_options(options)

def Facereconination():
    Ronaldo = cv2.imread("D:\\final-project\\face-reco\\images_test\\cristiano.jfif")
    Ronaldo = cv2.resize(Ronaldo, (512, 512))
    Ronaldo_face_location = fr.face_locations(Ronaldo[:,:,::-1])
    Ronaldo_face_encoding = fr.face_encodings(Ronaldo[:,:,::-1], Ronaldo_face_location)[0]
    # cv2.imshow("Ronaldo", Ronaldo)
    # t, r, b, l = Ronaldo_face_location[0]
    # cv2.imshow("Ronaldo-face", Ronaldo[t:b, l:r])

    messi = cv2.imread("D:\\final-project\\face-reco\\images_test\\messi.jfif")
    messi = cv2.resize(messi, (512, 512))
    messi_face_location = fr.face_locations(messi[:,:,::-1])
    messi_face_encoding = fr.face_encodings(messi[:,:,::-1], messi_face_location)[0]
    # cv2.imshow("Messi", messi)
    # t, r, b, l = messi_face_location[0]
    # cv2.imshow("Messi-face", messi[t:b, l:r])

    known_faces_encodings = [Ronaldo_face_encoding, messi_face_encoding]
    known_faces_names = ['Ronaldo', 'Messi']


    unknown = cv2.imread("D:\\final-project\\face-reco\\images_test\\messi-ronaldo.jfif")
    unknown = cv2.resize(unknown, (512, 512))
    unknown_face_locations = fr.face_locations(unknown[:,:,::-1])
    unknown_face_encodings = fr.face_encodings(unknown[:,:,::-1], unknown_face_locations)
    # cv2.imshow("unknown", unknown)
    # for i, (t, r, b, l) in enumerate(unknown_face_locations):
    #     cv2.imshow(f"unknown-face {i}", unknown[t:b, l:r])

    print(f"encoding {unknown_face_encodings}\n")
    

    tmp = unknown.copy()
    for (t, r, b, l), unknown_face_encoding in zip(unknown_face_locations, unknown_face_encodings):
        matches = fr.compare_faces(known_faces_encodings, unknown_face_encoding)
        print(f"{matches = }")
        # Instead, use face_distance to calculate similarities
        face_distances = fr.face_distance(known_faces_encodings, unknown_face_encoding)
        print(f"{face_distances = }")
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        if matches[best_match_index]:
            name = known_faces_names[best_match_index]
        # Draw a box around the face using the Pillow module
        tmp = cv2.rectangle(tmp, (l,t), (r,b), (255, 0, 0), 1)
        # Draw a label with a name below the face
        cv2.putText(tmp, name, (l,t +10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("after recongnition", tmp)
    cv2.imwrite("D:\\final-project\\face-reco\\images_test\\res2.png", tmp)
    cv2.waitKey(0)

def mediapipeWithFacereconination():

    Ronaldo = cv2.imread("D:\\final-project\\face-reco\\images_test\\cristiano.jfif")
    Ronaldo = cv2.resize(Ronaldo, (512, 512))
    mp_Ronaldo = mp.Image(image_format=mp.ImageFormat.SRGB, data=Ronaldo)
    face_detector_result = detector.detect(mp_Ronaldo)
    Ronaldo_face_image, Ronaldo_face_location, Ronaldo_prob_detect = detect_result(face_detector_result, Ronaldo, 20)
    Ronaldo_face_encoding = fr.face_encodings(Ronaldo_face_image[0][:,:,::-1], Ronaldo_face_location)
    cv2.imshow("Ronaldo", Ronaldo)
    cv2.imshow("Ronaldo-face", Ronaldo_face_image[0])

    messi = cv2.imread("D:\\final-project\\face-reco\\images_test\\messi.jfif")
    messi = cv2.resize(messi, (512, 512))
    mp_messi = mp.Image(image_format=mp.ImageFormat.SRGB, data=messi)
    face_detector_result = detector.detect(mp_messi)
    messi_face_image, messi_face_location, messi_prob_detect = detect_result(face_detector_result, messi, 20)
    messi_face_encoding = fr.face_encodings(messi_face_image[0][:,:,::-1], messi_face_location)
    cv2.imshow("Messi", messi)
    cv2.imshow("Messi-face", messi_face_image[0])

    known_faces_encodings = [Ronaldo_face_encoding, messi_face_encoding]
    known_faces_names = ['Ronaldo', 'Messi']


    unknown = cv2.imread("D:\\final-project\\face-reco\\images_test\\boten.jfif")
    unknown = cv2.resize(unknown, (512, 512))
    mp_unknown = mp.Image(image_format=mp.ImageFormat.SRGB, data=unknown)
    face_detector_result = detector.detect(mp_unknown)
    unknown_faces_images, locations_unknown, probs_unknown = detect_result(face_detector_result, unknown)
    encodings_unknown = fr.face_encodings(unknown[:,:,::-1], locations_unknown)
    cv2.imshow("UNKNOWN", unknown)
    cv2.imshow("unknown-face", unknown_faces_images[0])
    # t, r, b, l = locations_unknown[0]
    # cv2.rectangle(unknown, (t, l), (b,r), (255, 0, 0), 1)

    # tmp = unknown.copy()
    # for (t, r, b, l), encoding_unknown in zip(locations_unknown, encodings_unknown):
    #     matches = fr.compare_faces(known_faces_encodings, encoding_unknown)
    #     print(f"{matches = }")
    #     # Instead, use face_distance to calculate similarities
    #     face_distances = fr.face_distance(known_faces_encodings, encoding_unknown)
    #     best_match_index = np.argmin(face_distances)
    #     name = "Unknown"
    #     if matches[best_match_index]:
    #         name = known_faces_names[best_match_index]
    #     # Draw a box around the face using the Pillow module
    #     tmp = cv2.rectangle(tmp, (t, l), (b, r), (255, 0, 0), 1)
    #     # Draw a label with a name below the face
    #     cv2.putText(tmp, name, (t, l - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # cv2.imshow("after recongnition", tmp)
    # for i, unknown_face_image in enumerate(unknown_faces_images):
    #     cv2.imshow(f"face {i}", unknown_face_image)
    cv2.waitKey(0)

Facereconination()