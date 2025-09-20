import cv2
import mediapipe as mp
import os
import time
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "finger"
myList = os.listdir(folderPath)
overlayList = []
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

for im in myList:
    image = cv2.imread(f'{folderPath}/{im}')
    overlayList.append(image)
 
tipIds  = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) != 0:
        fingers = []
        
        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w,  c = overlayList[totalFingers-1].shape
        img[20:h+20, 20:w+20] = overlayList[totalFingers-1]
        cv2.rectangle(img, (20, 148), (170, 298), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 275), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0, 0), 25)
    else:
        cv2.rectangle(img, (20, 148), (170, 298), (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)