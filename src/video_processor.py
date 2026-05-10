import cv2
import os
import datetime

def draw_bounding_boxes(frame, detections, timestamp):
    # iterate over detections which are now ((x,y,w,h), label)
    for (box, label) in detections:
        x, y, w, h = box
        # draw the red rectangle for the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # draw a small filled rectangle for the label background
        cv2.rectangle(frame, (x, y - 20), (x + 100, y), (0, 0, 255), -1)
        # put the classification label (human, vehicle, etc)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # overlay the timestamp requested by the assignment brief
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

class VideoSaver:
    def __init__(self):
        self.writer = None
        self.output_folder = '../data/output_clips'
        # ensure output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def start_saving(self, frame):
        # generate filename based on current time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_folder, f"security_event_{timestamp}.avi")
        
        # multimedia tech: using xvid codec for avi container
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        height, width = frame.shape[:2]
        self.writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))

    def save_frame(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def stop_saving(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None