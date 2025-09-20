import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
                    static_image_mode=self.mode,
                    model_complexity=1,
                    smooth_landmarks=self.smooth,
                    enable_segmentation=False,
                    min_detection_confidence=self.detectionCon,
                    min_tracking_confidence=self.trackCon
                )


    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = lm.x*w, lm.y*h
                lmList.append([id, cx, cy]) 
                if draw:
                    cv2.circle(img, (int(cx), int(cy)), 5, (25, 35, 150), cv2.FILLED)
        return lmList 




def main():
    cap = cv2.VideoCapture('PoseVideos/6.mp4')
    pTime = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()