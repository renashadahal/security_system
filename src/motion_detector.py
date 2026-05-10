import cv2
import numpy as np

class MotionDetector:
    def __init__(self, first_frame):
        # Multimedia tech: convert to gray and blur to remove high-frequency noise
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.avg_background = cv2.GaussianBlur(gray, (25, 25), 0).astype("float")

    def detect_motion(self, current_frame, sensitivity_value):
        # Pre-process frame for computer vision analysis
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (25, 25), 0)

        # Update background model with a learning rate that catches distant movement
        cv2.accumulateWeighted(gray, self.avg_background, 0.05)
        background_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_background))

        # Thresholding to isolate motion pixels
        thresh = cv2.threshold(background_delta, sensitivity_value, 255, cv2.THRESH_BINARY)[1]
        
        # FIX: Increased iterations to 18 to merge fragmented boxes on vehicles
        thresh = cv2.dilate(thresh, None, iterations=18)

        # Find external contours of moving objects
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            # Catching distant humans requires a lower area threshold (approx 500)
            if cv2.contourArea(contour) < 500: 
                continue
                
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # REFINED HEURISTIC CATEGORIZATION
            if aspect_ratio < 0.75: 
                label = "Human"
            elif aspect_ratio > 1.3:
                label = "Vehicle"
            else:
                label = "Object"
                
            detections.append(((x, y, w, h), label))

        return detections, thresh