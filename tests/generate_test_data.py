# importing opencv, numpy, and os
import cv2
import numpy as np
import os

def generate_environments(input_filename):

    # getting the folder where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # moving one folder up to reach the main project folder
    project_root = os.path.dirname(script_dir)
    
    # making the full path for the input video
    input_path = os.path.join(
        project_root,
        "data",
        "input_videos",
        input_filename
    )

    # output folder path
    output_dir = os.path.join(project_root, "data")
    
    # checking if the input video exists
    if not os.path.exists(input_path):
        print(f"Error: Could not find input video at: {input_path}")
        return

    # opening the video
    cap = cv2.VideoCapture(input_path)

    # setting video format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # getting video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # output paths for generated videos
    path_b = os.path.join(output_dir, "env_b_variable.mp4")
    path_c = os.path.join(output_dir, "env_c_lowlight.mp4")

    # creating video writers
    out_b = cv2.VideoWriter(path_b, fourcc, fps, (width, height))
    out_c = cv2.VideoWriter(path_c, fourcc, fps, (width, height))

    # frame counter
    frame_count = 0

    # looping through the video
    while cap.isOpened():

        # reading frame
        ret, frame = cap.read()

        # stopping if video ends
        if not ret:
            break
        
        # ---------------- env b ----------------
        # creating changing brightness to simulate variable lighting

        brightness_factor = 1.0 + 0.4 * np.sin(frame_count / 15.0) 

        # applying brightness changes
        env_b_frame = cv2.convertScaleAbs(
            frame,
            alpha=brightness_factor,
            beta=10
        )
        
        # ---------------- env c ----------------
        # creating low light version with noise

        # darkening the frame but still keeping objects visible
        low_light = cv2.convertScaleAbs(
            frame,
            alpha=0.6,
            beta=20
        )
        
        # adding gaussian noise to make it look grainy
        noise = np.random.normal(
            0,
            5,
            low_light.shape
        ).astype(np.uint8)

        # combining noise with the dark frame
        env_c_frame = cv2.add(low_light, noise)

        # this line is repeated again accidentally
        env_c_frame = cv2.add(low_light, noise)

        # writing processed frames into output videos
        out_b.write(env_b_frame)
        out_c.write(env_c_frame)
        
        # increasing frame count
        frame_count += 1

        # printing progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Generating frame {frame_count}...")

    # closing video files
    cap.release()
    out_b.release()
    out_c.release()

    # showing success message
    print(f"\nSUCCESS! Files are located in your data folder:")

    print(f"-> {path_b}")
    print(f"-> {path_c}")

# starting point of the script
if __name__ == "__main__":

    # generating test environments using warehouse video
    generate_environments("warehouse1.mp4")