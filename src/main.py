# importing libraries
import tkinter as tk
from tkinter import ttk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import datetime

# importing the motion detector and video saving functions
from motion_detector import MotionDetector
from video_processor import draw_bounding_boxes, VideoSaver


class App:
    def __init__(self, root):
        self.root = root
        
        # setting up the app window
        self.root.title("Warehouse Smart Security System")
        self.root.geometry("1100x750")

        # making the detector and video saver
        self.detector = MotionDetector()
        self.saver = VideoSaver()

        # video related variables
        self.cap = None
        self.running = False
        self.is_recording = False
        
        # this keeps recording going for a bit after motion stops
        self.no_motion_frames = 0
        self.GRACE_PERIOD = 150 

        # finding the input video folder
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_folder = os.path.abspath(os.path.join(base_path, "..", "data", "input_videos"))

        # setting up the ui
        self.setup_ui()

        # loading video sources into the dropdown
        self.refresh_video_sources()

    def setup_ui(self):

        # left panel for buttons and logs
        self.side_panel = ttk.Frame(self.root, padding="15")
        self.side_panel.pack(side=tk.LEFT, fill=tk.Y)

        # dashboard title
        ttk.Label(self.side_panel, text="Security Dashboard", font=("Helvetica", 14, "bold")).pack(pady=(0, 20))

        # status section
        status_frame = ttk.LabelFrame(self.side_panel, text="System Status")
        status_frame.pack(fill=tk.X, pady=10)

        # shows the current system status
        self.status_label = ttk.Label(status_frame, text="● STANDBY", foreground="gray", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=5)

        # dropdown for selecting webcam or video
        ttk.Label(self.side_panel, text="Select Video Source:").pack(anchor=tk.W)
        self.source_var = tk.StringVar()
        self.source_dropdown = ttk.Combobox(self.side_panel, textvariable=self.source_var, state="readonly", width=25)
        self.source_dropdown.pack(pady=5)

        # start and stop buttons
        ttk.Button(self.side_panel, text="Start System", command=self.start_video).pack(fill=tk.X, pady=10)
        ttk.Button(self.side_panel, text="Stop / Reset", command=self.stop_video).pack(fill=tk.X, pady=5)

        # checkbox to show the motion mask
        self.view_mode_var = tk.BooleanVar()
        ttk.Checkbutton(self.side_panel, text="Show Motion Mask", variable=self.view_mode_var).pack(anchor=tk.W, pady=15)

        # log section
        ttk.Label(self.side_panel, text="System Log:").pack(anchor=tk.W)

        # text box for showing logs
        self.log_box = tk.Text(self.side_panel, width=28, height=12, font=("Consolas", 9), state="disabled")
        self.log_box.pack(pady=5)

        # area where the video will show up
        self.video_area = tk.Label(self.root, bg="black")
        self.video_area.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def log_message(self, message):

        # allowing text to be added
        self.log_box.config(state="normal")

        # adding the log message with the current time
        self.log_box.insert(tk.END, f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")

        # auto scrolling to the newest log
        self.log_box.see(tk.END)

        # disabling editing again
        self.log_box.config(state="disabled")

    def refresh_video_sources(self):

        # starting with webcam option
        sources = ["Webcam (Live)"]

        # adding video files from the folder
        if os.path.exists(self.input_folder):
            sources += [f for f in os.listdir(self.input_folder) if f.endswith((".mp4", ".avi", ".mkv"))]

        # putting the options into the dropdown
        self.source_dropdown["values"] = sources

        # selecting the first option automatically
        self.source_dropdown.current(0)

    def start_video(self):

        # stopping old video if already running
        if self.running: 
            self.stop_video()

        selected = self.source_var.get()

        # opening webcam if selected
        if selected == "Webcam (Live)":
            self.cap = cv2.VideoCapture(0)
            self.log_message("webcam started")

        else:
            # opening selected video file
            path = os.path.join(self.input_folder, selected)
            self.cap = cv2.VideoCapture(path)

            self.log_message(f"playing {selected}")

        # if video source doesnt open properly
        if not self.cap or not self.cap.isOpened():
            self.log_message("could not open source")
            return

        # starting the processing loop
        self.running = True
        self.process_loop()

    def stop_video(self):

        # stopping the system
        self.running = False

        # releasing the video source
        if self.cap: 
            self.cap.release()

        self.cap = None

        # stopping recording
        self.saver.stop_saving()
        self.is_recording = False

        # resetting the status label
        self.status_label.config(text="● STANDBY", foreground="gray")
        
        # making a black screen after stopping
        black = np.zeros((540, 960, 3), dtype=np.uint8)

        # converting image for tkinter
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(black))

        # showing the black screen
        self.video_area.img_tk = img_tk
        self.video_area.config(image=img_tk)

        self.log_message("system stopped")

    def process_loop(self):

        # stopping if system isnt running
        if not (self.running and self.cap): 
            return

        # reading the next frame
        ret, frame = self.cap.read()

        # stopping if frame cant be read
        if not ret:
            self.stop_video()
            return

        # resizing the frame
        frame = cv2.resize(frame, (960, 540))

        # checking for motion
        detections, thresh = self.detector.detect_motion(frame)

        # getting current time
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # drawing boxes around detected motion
        annotated = draw_bounding_boxes(frame.copy(), detections, timestamp)

        # if motion is detected
        if detections:

            # resetting no motion counter
            self.no_motion_frames = 0

            # checking if any detected object is a human
            has_human = any(label == "Human" for _, label in detections)

            # changing the status label
            self.status_label.config(
                text="● ALERT" if has_human else "● MOTION",
                foreground="red" if has_human else "orange"
            )

            # starting recording if not already recording
            if not self.is_recording:

                self.saver.start_saving(annotated)

                # saving a screenshot of the first detection
                self.saver.save_snapshot(annotated)

                self.is_recording = True
                self.log_message("recording triggered")

            # saving the current frame
            self.saver.save_frame(annotated)

        else:

            # showing active status when no motion is found
            self.status_label.config(text="● ACTIVE", foreground="green")

            # if recording is still active
            if self.is_recording:

                self.no_motion_frames += 1

                # stopping recording after enough no-motion frames
                if self.no_motion_frames >= self.GRACE_PERIOD:
                    self.saver.stop_saving()
                    self.is_recording = False

                    self.log_message("recording ended")

                else:
                    # still saving frames during cooldown
                    self.saver.save_frame(annotated)

        # showing mask view or normal view
        display = thresh if self.view_mode_var.get() else annotated

        # choosing the right color conversion
        color = cv2.COLOR_GRAY2RGB if display.ndim == 2 else cv2.COLOR_BGR2RGB

        # converting colors for tkinter
        display_rgb = cv2.cvtColor(display, color)
        
        # converting image into tkinter format
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(display_rgb))

        # updating the video display
        self.video_area.img_tk = img_tk
        self.video_area.config(image=img_tk)

        # repeating the loop every 30ms
        self.root.after(30, self.process_loop)


# starting the app
if __name__ == "__main__":

    # making the main window
    root = tk.Tk()

    # creating the app
    app = App(root)

    # keeping the app running
    root.mainloop()