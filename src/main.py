# main.py

import tkinter as tk
from tkinter import ttk
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import datetime

from motion_detector import MotionDetector
from video_processor import draw_bounding_boxes, VideoSaver


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Warehouse Smart Security System")
        self.root.geometry("1100x750")

        self.detector = MotionDetector()
        self.saver = VideoSaver()

        self.cap = None
        self.running = False
        self.is_recording = False
        
        # Cooldown prevents the recording from cutting out mid-event (approx 5 seconds).
        self.no_motion_frames = 0
        self.GRACE_PERIOD = 150 

        base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_folder = os.path.abspath(os.path.join(base_path, "..", "data", "input_videos"))

        self.setup_ui()
        self.refresh_video_sources()

    def setup_ui(self):
        self.side_panel = ttk.Frame(self.root, padding="15")
        self.side_panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(self.side_panel, text="Security Dashboard", font=("Helvetica", 14, "bold")).pack(pady=(0, 20))

        status_frame = ttk.LabelFrame(self.side_panel, text="System Status")
        status_frame.pack(fill=tk.X, pady=10)

        self.status_label = ttk.Label(status_frame, text="● STANDBY", foreground="gray", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=5)

        ttk.Label(self.side_panel, text="Select Video Source:").pack(anchor=tk.W)
        self.source_var = tk.StringVar()
        self.source_dropdown = ttk.Combobox(self.side_panel, textvariable=self.source_var, state="readonly", width=25)
        self.source_dropdown.pack(pady=5)

        ttk.Button(self.side_panel, text="Start System", command=self.start_video).pack(fill=tk.X, pady=10)
        ttk.Button(self.side_panel, text="Stop / Reset", command=self.stop_video).pack(fill=tk.X, pady=5)

        self.view_mode_var = tk.BooleanVar()
        ttk.Checkbutton(self.side_panel, text="Show Motion Mask", variable=self.view_mode_var).pack(anchor=tk.W, pady=15)

        ttk.Label(self.side_panel, text="System Log:").pack(anchor=tk.W)
        self.log_box = tk.Text(self.side_panel, width=28, height=12, font=("Consolas", 9), state="disabled")
        self.log_box.pack(pady=5)

        self.video_area = tk.Label(self.root, bg="black")
        self.video_area.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def log_message(self, message):
        self.log_box.config(state="normal")
        self.log_box.insert(tk.END, f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state="disabled")

    def refresh_video_sources(self):
        sources = ["Webcam (Live)"]
        if os.path.exists(self.input_folder):
            sources += [f for f in os.listdir(self.input_folder) if f.endswith((".mp4", ".avi", ".mkv"))]
        self.source_dropdown["values"] = sources
        self.source_dropdown.current(0)

    def start_video(self):
        if self.running: self.stop_video()
        selected = self.source_var.get()
        if selected == "Webcam (Live)":
            self.cap = cv2.VideoCapture(0)
            self.log_message("webcam started")
        else:
            path = os.path.join(self.input_folder, selected)
            self.cap = cv2.VideoCapture(path)
            self.log_message(f"playing {selected}")

        if not self.cap or not self.cap.isOpened():
            self.log_message("could not open source")
            return
        self.running = True
        self.process_loop()

    def stop_video(self):
        self.running = False
        if self.cap: self.cap.release()
        self.cap = None
        self.saver.stop_saving()
        self.is_recording = False
        self.status_label.config(text="● STANDBY", foreground="gray")
        
        black = np.zeros((540, 960, 3), dtype=np.uint8)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(black))
        self.video_area.img_tk = img_tk
        self.video_area.config(image=img_tk)
        self.log_message("system stopped")

    def process_loop(self):
        if not (self.running and self.cap): return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        frame = cv2.resize(frame, (960, 540))
        detections, thresh = self.detector.detect_motion(frame)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        annotated = draw_bounding_boxes(frame.copy(), detections, timestamp)

        if detections:
            self.no_motion_frames = 0
            has_human = any(label == "Human" for _, label in detections)
            self.status_label.config(
                text="● ALERT" if has_human else "● MOTION",
                foreground="red" if has_human else "orange"
            )

            if not self.is_recording:
                self.saver.start_saving(annotated)
                self.saver.save_snapshot(annotated) # Save image of initial trigger.
                self.is_recording = True
                self.log_message("recording triggered")
            self.saver.save_frame(annotated)
        else:
            self.status_label.config(text="● ACTIVE", foreground="green")
            if self.is_recording:
                self.no_motion_frames += 1
                if self.no_motion_frames >= self.GRACE_PERIOD:
                    self.saver.stop_saving()
                    self.is_recording = False
                    self.log_message("recording ended")
                else:
                    self.saver.save_frame(annotated)

        display = thresh if self.view_mode_var.get() else annotated
        color = cv2.COLOR_GRAY2RGB if display.ndim == 2 else cv2.COLOR_BGR2RGB
        display_rgb = cv2.cvtColor(display, color)
        
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(display_rgb))
        self.video_area.img_tk = img_tk
        self.video_area.config(image=img_tk)
        self.root.after(30, self.process_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()