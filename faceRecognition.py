import numpy as np 
import os 
from keras_facenet import FaceNet
import pickle
import cv2
from PIL import Image
import mediapipe as mp
import time
from tensorflow.keras.models import load_model



facenet = FaceNet()
faceDetector = mp.solutions.face_detection
faceDetection = faceDetector.FaceDetection()

myFile = open("signatureData.pkl", "rb")
signatureDatabase = pickle.load(myFile)
myFile.close()

cap = cv2.VideoCapture(0)
cTime, pTime = 0, 0

model = load_model("Face Mask Detection Model.h5")


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
    # else:
    #     bbox = [1, 1, 10, 10]

            face = imgRgb[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            face = Image.fromarray(face)
            face = face.resize((160, 160))
            face = np.asarray(face)
            face = np.expand_dims(face, axis=0)
            signature = facenet.embeddings(face)

            minDist=100
            identity=' '
            for key, value in signatureDatabase.items() :
                dist = np.linalg.norm(value-signature)
                if dist < minDist:
                    minDist = dist
                    identity = key
            
            cv2.putText(img,identity, (bbox[0]-10, bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cTime = time.time()
    fps = str(int(1 / (cTime-pTime)))
    pTime = cTime
    cv2.putText(img,fps, (200,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Image',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
cv2.destroyAllWindows()
cap.release()