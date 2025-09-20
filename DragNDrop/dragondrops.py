import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
colorR = (0, 0, 255)

# Rectangle class
class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # Check if cursor is within rectangle area
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

# Create multiple draggable rectangles
rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror effect

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]['lmList']

        if len(lmList) >= 13:
            point1 = lmList[8][:2]   # Index fingertip (x, y)
            point2 = lmList[12][:2]  # Middle fingertip (x, y)

            # Calculate distance between index and middle fingertip
            length, _, _ = detector.findDistance(point1, point2)
            print("Pinch distance:", length)

            if length < 40:
                cursor = point1
                for rect in rectList:
                    rect.update(cursor)

    # Create transparent layer
    imgNew = np.zeros_like(img, np.uint8)

    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Blend transparent layer with original image
    out = img.copy()
    alpha = 0.1
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Show output
    cv2.imshow("Image", out)
    cv2.waitKey(1)
