# importing opencv, os, and time/date stuff
import cv2
import os
import datetime

def draw_bounding_boxes(frame, detections, timestamp):

    # going through all detected objects
    for (box, label) in detections:

        # getting box position and size
        x, y, w, h = box

        # drawing red rectangle around detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # getting size of the label text
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        # making sure the label stays inside the frame
        label_top = max(y - text_h - baseline - 6, 0)

        # drawing red background for the label
        cv2.rectangle(
            frame,
            (x, label_top),
            (x + text_w + 6, y),
            (0, 0, 255),
            -1
        )

        # putting the object label text on the frame
        cv2.putText(
            frame,
            label,
            (x + 3, y - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    # showing timestamp in the top left corner
    cv2.putText(
        frame,
        timestamp,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # returning the edited frame
    return frame


class VideoSaver:
    def __init__(self):

        # video writer starts as none
        self.writer = None

        # getting the main project folder path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # folder for saved video clips
        self.output_folder = os.path.join(base_dir, "data", "output_clips")

        # folder for saved screenshots
        self.snapshot_folder = os.path.join(base_dir, "data", "event_snapshots")

        # creating folders if they dont already exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.snapshot_folder, exist_ok=True)

    def start_saving(self, frame):

        # making a timestamp for the filename
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # making the video filename
        filename = os.path.join(
            self.output_folder,
            f"security_event_{ts}.avi"
        )

        # setting video format
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # getting frame size
        h, w = frame.shape[:2]

        # creating the video writer
        self.writer = cv2.VideoWriter(
            filename,
            fourcc,
            20.0,
            (w, h)
        )

    def save_snapshot(self, frame):

        # making timestamp for snapshot name
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # creating snapshot file path
        path = os.path.join(
            self.snapshot_folder,
            f"trigger_{ts}.jpg"
        )

        # saving image file
        cv2.imwrite(path, frame)

    def save_frame(self, frame):

        # saving frame into the video if writer exists
        if self.writer is not None:
            self.writer.write(frame)

    def stop_saving(self):

        # stopping and closing the video file
        if self.writer is not None:
            self.writer.release()

            # resetting writer back to none
            self.writer = None