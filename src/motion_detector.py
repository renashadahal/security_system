# motion_detector.py
import cv2
import numpy as np

class MotionDetector:
    def __init__(self):
        # MOG2 background subtraction with shadow detection
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=True
        )
        
        # Lighter kernels to avoid erasing small human-sized motion blobs
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Buffer to store previous frame detections for stability
        self.previous_boxes = []

    def boxes_similar(self, box1, box2):
        """Calculates distance between centroids to track objects across frames."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        distance = abs(cx1 - cx2) + abs(cy1 - cy2)
        return distance < 45

    def detect_motion(self, current_frame):
        # Pre-processing: Grayscale and Blur to reduce camera noise
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Low learningRate (0.0001) prevents slow-moving humans from being absorbed into background
        fg_mask = self.back_sub.apply(gray, learningRate=0.0001)
        
        # Thresholding to keep only high-confidence motion pixels
        thresh = cv2.inRange(fg_mask, 250, 255)
        
        # Lighter noise reduction for distant/small objects
        thresh = cv2.medianBlur(thresh, 3) 
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.open_kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.close_kernel)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Identify distinct motion areas
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        h_f, w_f = current_frame.shape[:2]

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Lowered area threshold (80) to capture distant figures
            if area < 80 or area > (h_f * w_f * 0.7):
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Heuristic classification based on warehouse perspective
            if 0.2 < aspect_ratio < 1.0 and h > 25:
                label = "Human"
            elif aspect_ratio >= 1.2 and w > 50:
                label = "Vehicle"
            else:
                label = "Object"
            
            detections.append(((x, y, w, h), label))

        # Filter out random flicker: only show objects seen in consecutive frames
        stable_detections = []
        for det in detections:
            if any(self.boxes_similar(det[0], pb) for pb in self.previous_boxes):
                stable_detections.append(det)
        
        self.previous_boxes = [box for box, label in detections]
        return stable_detections, thresh