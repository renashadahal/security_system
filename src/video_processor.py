# video_processor.py
import cv2
import os
import datetime

def draw_bounding_boxes(frame, detections, timestamp):
    for (box, label) in detections:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Labeling logic
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(y - th - bl - 6, 0)
        cv2.rectangle(frame, (x, label_top), (x + tw + 6, y), (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 3, y - bl - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

class VideoSaver:
    def __init__(self):
        self.writer = None
        # Absolute paths ensure data folders are created correctly
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_folder = os.path.join(base_dir, "data", "output_clips")
        self.snapshot_folder = os.path.join(base_dir, "data", "event_snapshots")

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.snapshot_folder, exist_ok=True)

    def start_saving(self, frame):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_folder, f"security_event_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        h, w = frame.shape[:2]
        self.writer = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))

    def save_snapshot(self, frame):
        """Saves a high-quality JPG image of the initial motion trigger."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.snapshot_folder, f"alert_{ts}.jpg")
        cv2.imwrite(path, frame)

    def save_frame(self, frame):
        if self.writer:
            self.writer.write(frame)

    def stop_saving(self):
        if self.writer:
            self.writer.release()
            self.writer = None