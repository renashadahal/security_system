# importing opencv and numpy
import cv2
import numpy as np

class MotionDetector:
    def __init__(self):
        # making the background subtractor
        # history=500 helps it remember the background for about 20 seconds at 25fps
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=True
        )

        # small kernel for removing tiny noise dots
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        # BIGGER KERNEL: Increased to (15, 15) to merge fragmented boxes
        # This acts like a "glue" to connect motion pixels that are close together
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (15, 15)
        )

        # saving previous detection boxes for stability checking
        self.previous_boxes = []

    def boxes_similar(self, box1, box2):
        # getting values from both boxes
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # finding center points
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

        # Manhattan distance check
        distance = abs(cx1 - cx2) + abs(cy1 - cy2)

        # threshold for stability; if boxes are close across frames, it's a "real" object
        return distance < 50

    def detect_motion(self, current_frame):
        # turning frame into grayscale and blurring to remove sensor noise
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # applying background subtraction
        fg_mask = self.back_sub.apply(
            gray,
            learningRate=0.01
        )

        # keep only strong white motion areas (removes gray shadows)
        thresh = cv2.inRange(fg_mask, 240, 255)

        # 1. MORPH_OPEN: Removes tiny "salt" noise (white dots)
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            self.open_kernel,
            iterations=1
        )

        # 2. DILATE: Makes white motion areas fatter to bridge gaps
        # Increased iterations to 3 for better connectivity
        thresh = cv2.dilate(thresh, None, iterations=3)

        # 3. MORPH_CLOSE: Fills the holes inside the blobs
        # Uses the larger 15x15 kernel to snap fragmented boxes into one
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            self.close_kernel,
            iterations=2
        )

        # finding outlines around the merged motion blobs
        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        h_frame, w_frame = current_frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)

            # Ignore tiny noise; minimum area increased to avoid flickering small boxes
            if area < 500 or area > (h_frame * w_frame * 0.7):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Heuristic classification
            if 0.2 < aspect_ratio < 1.2 and h > 40:
                label = "Human"
            elif aspect_ratio >= 1.2 and w > 60:
                label = "Vehicle"
            else:
                label = "Object"

            detections.append(((x, y, w, h), label))

        # Stability logic: ensures boxes are persistent across frames
        stable_detections = []
        for det in detections:
            if not self.previous_boxes:
                stable_detections.append(det)
                continue
            
            if any(self.boxes_similar(det[0], pb) for pb in self.previous_boxes):
                stable_detections.append(det)

        self.previous_boxes = [box for box, _ in detections]

        return stable_detections, thresh