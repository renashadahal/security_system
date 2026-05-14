# importing time, opencv, and system stuff
import time
import cv2
import os
import sys

# adding the main project folder to python's path
# this lets the script import files from the project
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

# trying to import the main security system class
try:
    from src.main import SecuritySystem 

# showing error if the import doesnt work
except ImportError:
    print("Error: Could not find SecuritySystem in src/main.py. Check your filenames!")

def run_speed_test(video_path):

    # checking if the video file exists
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    # opening the video
    cap = cv2.VideoCapture(video_path)
    
    # this is where the real system could be initialized
    # app = SecuritySystem() 

    # starting timer
    start_time = time.time()

    # counting how many frames get processed
    frame_count = 0

    print("Starting Benchmark...")

    # looping through video frames
    while cap.isOpened():

        # reading frame
        ret, frame = cap.read()

        # stopping if video ends or 300 frames are processed
        if not ret or frame_count > 300: 
            break
        
        # converting frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # blurring frame to simulate processing workload
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # increasing frame counter
        frame_count += 1

    # ending timer
    end_time = time.time()

    # total processing time
    total_time = end_time - start_time

    # calculating fps
    fps = frame_count / total_time

    # calculating average latency per frame
    avg_latency = (total_time / frame_count) * 1000

    # printing benchmark results
    print("-" * 30)
    print(f"RESULTS FOR: {os.path.basename(video_path)}")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Average FPS: {fps:.2f}")
    print(f"Avg Latency per Frame: {avg_latency:.2f} ms")
    print("-" * 30)

# starting point of the script
if __name__ == "__main__":

    # running speed test on the lowlight video
    run_speed_test("data/input_videos/env_c_lowlight.mp4")