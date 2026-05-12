# motion_detector.py

import cv2
import numpy as np


class MotionDetector:
    def __init__(self):
        # Increased sensitivity (varThreshold=25) to reduce "phantom" detections 
        # on static objects while staying responsive.
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=25,
            detectShadows=True
        )

        # Smaller kernels prevent the system from "erasing" small human blobs.
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2, 2)
        )

        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        # Stores previous frame detections to ensure motion is consistent.
        self.previous_boxes = []

    def boxes_similar(self, box1, box2):
        """Checks if two boxes in consecutive frames are roughly in the same spot."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        cx1 = x1 + w1 // 2
        cy1 = y1 + h1 // 2

        cx2 = x2 + w2 // 2
        cy2 = y2 + h2 // 2

        distance = abs(cx1 - cx2) + abs(cy1 - cy2)
        return distance < 50

    def detect_motion(self, current_frame):
        # Pre-processing
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Learning rate set to 0.01 to adapt to warehouse lighting without losing people.
        fg_mask = self.back_sub.apply(
            gray,
            learningRate=0.01
        )

        # Keep strong foreground pixels and remove shadows.
        thresh = cv2.inRange(fg_mask, 240, 255)

        # Lighter noise reduction (3x3) so the tracker sees what the mask sees.
        thresh = cv2.medianBlur(thresh, 3)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            self.open_kernel,
            iterations=1
        )

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            self.close_kernel,
            iterations=1
        )

        # Slightly expand detections to connect fragmented motion.
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        h_frame, w_frame = current_frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)

            # Lowered area threshold (60) to catch distant/small human figures.
            if area < 60 or area > (h_frame * w_frame * 0.7):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Broadened classification for high camera angles and perspective.
            if 0.2 < aspect_ratio < 1.2 and h > 25:
                label = "Human"
            elif aspect_ratio >= 1.2 and w > 50:
                label = "Vehicle"
            else:
                label = "Object"

            detections.append(((x, y, w, h), label))

        # Stability check: ensure motion appears in consecutive frames.
        stable_detections = []
        for det in detections:
            if not self.previous_boxes:
                stable_detections.append(det)
                continue
                
            if any(self.boxes_similar(det[0], pb) for pb in self.previous_boxes):
                stable_detections.append(det)

        self.previous_boxes = [box for box, _ in detections]
        return stable_detections, thresh