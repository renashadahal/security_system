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
        self.root.geometry("1150x750")

        self.detector = MotionDetector()
        self.saver = VideoSaver()

        self.cap = None
        self.running = False
        self.is_recording = False
        
        # Recording Cooldown Logic (~5 seconds at 30 FPS)
        self.no_motion_frames = 0
        self.RECORDING_GRACE_PERIOD = 150 

        # Setup Absolute Paths for Data Management
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.input_folder = os.path.join(base_dir, "data", "input_videos")
        self.log_folder = os.path.join(base_dir, "data", "system_logs")
        
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)

        self.setup_ui()
        self.refresh_sources()

    def setup_ui(self):
        self.side_panel = ttk.Frame(self.root, padding="15")
        self.side_panel.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(self.side_panel, text="Security Dashboard", font=("Helvetica", 14, "bold")).pack(pady=(0, 20))
        self.status_label = ttk.Label(self.side_panel, text="● STANDBY", foreground="gray", font=("Helvetica", 10, "bold"))
        self.status_label.pack(pady=10)

        ttk.Label(self.side_panel, text="Video Source:").pack(anchor=tk.W)
        self.source_var = tk.StringVar()
        self.source_dropdown = ttk.Combobox(self.side_panel, textvariable=self.source_var, state="readonly", width=25)
        self.source_dropdown.pack(pady=5)

        ttk.Button(self.side_panel, text="Start System", command=self.start_video).pack(fill=tk.X, pady=10)
        ttk.Button(self.side_panel, text="Stop & Save Logs", command=self.stop_video).pack(fill=tk.X, pady=5)

        self.view_mode_var = tk.BooleanVar()
        ttk.Checkbutton(self.side_panel, text="Show Motion Mask", variable=self.view_mode_var).pack(anchor=tk.W, pady=15)

        ttk.Label(self.side_panel, text="System Log:").pack(anchor=tk.W)
        self.log_box = tk.Text(self.side_panel, width=30, height=15, font=("Consolas", 8), state="disabled")
        self.log_box.pack(pady=5)

        self.video_area = tk.Label(self.root, bg="black")
        self.video_area.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def log_message(self, message):
        self.log_box.config(state="normal")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{ts}] {message}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state="disabled")

    def save_logs_to_file(self):
        """Archives the UI log box content to a text file."""
        log_content = self.log_box.get("1.0", tk.END).strip()
        if not log_content: return
        
        path = os.path.join(self.log_folder, f"log_{datetime.date.today()}.txt")
        with open(path, "a") as f:
            f.write(f"\n--- SESSION: {datetime.datetime.now()} ---\n{log_content}\n")

    def refresh_sources(self):
        sources = ["Webcam 0"]
        if os.path.exists(self.input_folder):
            sources += [f for f in os.listdir(self.input_folder) if f.endswith(('.mp4', '.avi'))]
        self.source_dropdown['values'] = sources
        self.source_dropdown.current(0)

    def start_video(self):
        if self.running: self.stop_video()
        src = self.source_var.get()
        self.cap = cv2.VideoCapture(0 if src == "Webcam 0" else os.path.join(self.input_folder, src))
        self.running = True
        self.log_message(f"System Active: {src}")
        self.process_loop()

    def stop_video(self):
        self.running = False
        if self.cap: self.cap.release()
        self.save_logs_to_file()
        self.saver.stop_saving()
        self.is_recording = False
        self.status_label.config(text="● STANDBY", foreground="gray")
        self.log_message("System stopped. Logs archived.")
        
        black = np.zeros((540, 960, 3), dtype=np.uint8)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(black))
        self.video_area.img_tk = img_tk
        self.video_area.config(image=img_tk)

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
            self.status_label.config(text="● ALERT", foreground="red")

            if not self.is_recording:
                self.saver.start_saving(annotated)
                self.saver.save_snapshot(annotated) 
                self.is_recording = True
                self.log_message("Recording Triggered")

            self.saver.save_frame(annotated)
        else:
            self.status_label.config(text="● ACTIVE", foreground="green")
            if self.is_recording:
                self.no_motion_frames += 1
                if self.no_motion_frames >= self.RECORDING_GRACE_PERIOD:
                    self.saver.stop_saving()
                    self.is_recording = False
                    self.log_message("Recording Finished")
                else:
                    self.saver.save_frame(annotated)

        display = thresh if self.view_mode_var.get() else annotated
        color_mode = cv2.COLOR_GRAY2RGB if display.ndim == 2 else cv2.COLOR_BGR2RGB
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display, color_mode)))
        self.video_area.img_tk = img_tk
        self.video_area.config(image=img_tk)
        
        self.root.after(30, self.process_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()