import numpy as np 
import os 
from keras_facenet import FaceNet
import pickle
import cv2
from PIL import Image
import mediapipe as mp


facenet = FaceNet()
faceDetector = mp.solutions.face_detection
faceDetection = faceDetector.FaceDetection()

imgFolder = "D:\Secured Home\SecuredHome\Project Files\ImageDatabase"
signatureDatabase = {}

for filename in os.listdir(imgFolder):
    path = os.path.join(imgFolder, filename)
    img = cv2.imread(path)
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRgb = Image.fromarray(imgRgb)
    imgRgb = np.asarray(imgRgb)
    results = faceDetection.process(imgRgb)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bbox_class = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bbox_class.xmin * w), int(bbox_class.ymin * h), int(bbox_class.width * w), int(bbox_class.height * h)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0))
    else:
        bbox = [1, 1, 10, 10]

    face = imgRgb[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    face = Image.fromarray(face)
    face = face.resize((160, 160))
    face = np.asarray(face)
    face = np.expand_dims(face, axis=0)
    signature = facenet.embeddings(face)
    signatureDatabase[os.path.splitext(filename)[0]] = signature


print(signatureDatabase)

myFile = open("signatureData.pkl", "wb")
pickle.dump(signatureDatabase, myFile)
myFile.close()