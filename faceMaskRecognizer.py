import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
model = load_model("D:\Secured Home\SecuredHome\Project Files\Face Mask Detection Model.h5")
faceDetector = mp.solutions.face_detection
faceDetection = faceDetector.FaceDetection()
categories = {0: 'Mask', 1: 'No Mask'}
cTime, pTime = 0,0

while True:
    _, img = cap.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRgb)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bbox_class = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), int(bbox_class.width * w), int(bbox_class.height * h)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0))

            face = imgRgb[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            face = cv2.resize(face, (224, 224))
            imgA = np.asarray(face)
            nImgA = (imgA.astype(np.float32) / 127 -1)
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = nImgA
            prediction = model.predict(data)
            category = np.argmax(prediction[0])
            percentage = max(prediction[0]) * 100
            cv2.putText(img, categories[category], (bbox[0], bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            
    cTime = time.time()
    fps = str(int(1 / (cTime-pTime)))
    pTime = cTime
    cv2.putText(img,fps, (200,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
cv2.destroyAllWindows()
cap.release()