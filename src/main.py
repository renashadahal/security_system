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
        self.root.geometry("1000x700")
        
        self.saver = VideoSaver()
        self.detector = None
        self.is_recording = False
        
        # Fix paths to find the data folder from the src directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.input_folder = os.path.abspath(os.path.join(base_path, '..', 'data', 'input_videos'))

        # Modern GUI Layout Panels
        self.side_panel = ttk.Frame(self.root, padding="15")
        self.side_panel.pack(side=tk.LEFT, fill=tk.Y)

        header = ttk.Label(self.side_panel, text="Security Dashboard", font=('Helvetica', 14, 'bold'))
        header.pack(pady=(0, 20))

        # Functional Status Indicator
        self.status_frame = ttk.LabelFrame(self.side_panel, text="System Status")
        self.status_frame.pack(fill=tk.X, pady=10)
        self.status_label = ttk.Label(self.status_frame, text="● STANDBY", foreground="gray", font=('Helvetica', 10, 'bold'))
        self.status_label.pack(pady=5)

        ttk.Label(self.side_panel, text="Select Video Source:").pack(anchor=tk.W)
        self.source_var = tk.StringVar()
        self.source_dropdown = ttk.Combobox(self.side_panel, textvariable=self.source_var, state="readonly", width=22)
        self.source_dropdown.pack(pady=5)

        self.start_button = ttk.Button(self.side_panel, text="Start System", command=self.start_video)
        self.start_button.pack(fill=tk.X, pady=10)

        self.stop_button = ttk.Button(self.side_panel, text="Stop / Reset", command=self.stop_video)
        self.stop_button.pack(fill=tk.X, pady=5)

        ttk.Label(self.side_panel, text="Sensitivity:").pack(anchor=tk.W, pady=(15, 2))
        self.sensitivity_slider = ttk.Scale(self.side_panel, from_=5, to=150, orient=tk.HORIZONTAL)
        self.sensitivity_slider.set(25)
        self.sensitivity_slider.pack(fill=tk.X, pady=5)

        self.view_mode_var = tk.BooleanVar()
        self.view_mode_checkbox = ttk.Checkbutton(self.side_panel, text="Show Motion Mask", variable=self.view_mode_var)
        self.view_mode_checkbox.pack(anchor=tk.W, pady=15)

        ttk.Label(self.side_panel, text="System Log:").pack(anchor=tk.W)
        self.log_box = tk.Text(self.side_panel, width=25, height=10, font=('Consolas', 9), state='disabled')
        self.log_box.pack(pady=5)

        self.refresh_video_sources()

        self.video_area = tk.Label(self.root, bg="black")
        self.video_area.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.cap = None
        self.running = False

    def log_message(self, message):
        self.log_box.config(state='normal')
        self.log_box.insert(tk.END, f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state='disabled')

    def refresh_video_sources(self):
        sources = ["Webcam (Live)"]
        if os.path.exists(self.input_folder):
            video_files = [f for f in os.listdir(self.input_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
            sources.extend(video_files)
        self.source_dropdown['values'] = sources
        self.source_dropdown.current(0)

    def start_video(self):
        if self.running: self.stop_video()
            
        selected_source = self.source_var.get()
        if selected_source == "Webcam (Live)":
            self.cap = cv2.VideoCapture(0)
            self.log_message("Webcam mode activated.")
        else:
            video_path = os.path.join(self.input_folder, selected_source)
            self.cap = cv2.VideoCapture(video_path)
            self.log_message(f"Playing: {selected_source}")

        if not self.cap or not self.cap.isOpened():
            self.log_message("Error: Could not open source.")
            return

        self.running = True
        self.status_label.config(text="● ACTIVE", foreground="green")
        self.detector = None 
        self.process_loop()

    def stop_video(self):
        self.running = False
        if self.cap: self.cap.release()
        self.cap = None
        self.saver.stop_saving()
        self.is_recording = False
        self.status_label.config(text="● STANDBY", foreground="gray")
        
        # UI Requirement: Reset to black screen
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(black_frame))
        self.video_area.img_tk = img_tk
        self.video_area.config(image=img_tk)
        self.log_message("System reset.")

    def process_loop(self):
        if self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Performance fix: Resize large inputs to 640x480
                frame = cv2.resize(frame, (640, 480))
                
                if self.detector is None:
                    self.detector = MotionDetector(frame)

                labeled_boxes, thresh = self.detector.detect_motion(frame, int(self.sensitivity_slider.get()))
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                annotated_frame = draw_bounding_boxes(frame.copy(), labeled_boxes, timestamp)

                if len(labeled_boxes) > 0:
                    has_human = any(lbl == "Human" for _, lbl in labeled_boxes)
                    if has_human:
                        self.status_label.config(text="● ALERT", foreground="red")
                    
                    if not self.is_recording:
                        self.saver.start_saving(annotated_frame)
                        self.is_recording = True
                        msg = "ALERT: Unauthorized Access!" if has_human else "Motion detected."
                        self.log_message(msg)
                    self.saver.save_frame(annotated_frame)
                else:
                    self.status_label.config(text="● ACTIVE", foreground="green")
                    if self.is_recording:
                        self.saver.stop_saving()
                        self.is_recording = False

                # GUI Update Logic
                display = thresh if self.view_mode_var.get() else annotated_frame
                if len(display.shape) == 2:
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)
                else:
                    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                img_tk = ImageTk.PhotoImage(image=Image.fromarray(display))
                self.video_area.img_tk = img_tk
                self.video_area.config(image=img_tk)
                
                # Slower after(30) to reduce processing lag
                self.root.after(30, self.process_loop)
            else:
                self.stop_video()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()