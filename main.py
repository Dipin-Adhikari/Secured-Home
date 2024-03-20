import cv2
import mediapipe as mp
import time


PATH = r"D:\Secured Home\SecuredHome\Project Files\Video\robbery2.mp4"
cap = cv2.VideoCapture(PATH)
mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
cTime, pTime = 0, 0


while True:
    ret, img = cap.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRgb)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    cTime = time.time()
    fps = int(1 / (cTime - pTime))
    pTime = cTime
    cv2.putText(img, str(fps), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Camera 1", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
