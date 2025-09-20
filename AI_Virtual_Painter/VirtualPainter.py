import cv2
import mediapipe as mp
import numpy as np
import os
import HandTrackingModule as htm

folderPath = "Header"
myList = os.listdir(folderPath)

brushThickness = 15
canvas = np.zeros((720, 1250, 3), np.uint8)
overlayList = []
for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    if image is not None:
        image = cv2.resize(image, (1250, 125))
        overlayList.append(image)

header = overlayList[0] if overlayList else np.zeros((125, 1250, 3), dtype=np.uint8)

detector = htm.handDetector(detectionCon=0.8)

cap = cv2.VideoCapture(0)
cap.set(3, 1250)
cap.set(4, 720)

drawColor = (0, 0, 255)
xp, yp = 0, 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    img[0:125, 0:1250] = header
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            print("Selection mode")

            if y1 < 125:
                if 250 < x1 < 500:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 500 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 750 < x1 < 1000:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1000 < x1 < 1250:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1,y1-15), (x2, y2+15), drawColor, cv2.FILLED)

        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 5, drawColor, brushThickness)
            print("Drawing mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(canvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1


    img[0:125, 0:1250] = header
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

cap.release()
cv2.destroyAllWindows()
