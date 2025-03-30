import cv2
import numpy as np
import os
from pathlib import Path
import shutil

def find_project_root(marker=".gitignore"):
    """
    walk up from the current working directory until a directory containing the
    specified marker (e.g., .gitignore) is found.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}"
    )

def upscale_image(image, scale_factor=2):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def detect_chessboard(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        # more iterations = improve accuracy 
        # lower epsilon = higher precision
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
        # more window size = less accurate but detect more frames
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

def select_uniform_frames(frames_list, n):
    """
    selects a fixed number of frames evenly from the list of good frames found
    """
    if len(frames_list) <= n:
        return frames_list
    indices = np.linspace(0, len(frames_list) - 1, n, dtype=int)
    return [frames_list[i] for i in indices]

def process_video(
    record_name, 
    video_path, 
    output_dir, 
    pattern_size, 
    num_frames, 
    skip_seconds=1):
    # clean the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = int(total_frames / fps)

    detected_frames = []
    
    # for every second in the video (skipping seconds as specified)
    for sec in range(0, duration_seconds, skip_seconds):
        frame_no = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break
        
        if record_name == "CAM_2" or record_name == "CAM_3":
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
        found, corners = detect_chessboard(frame, pattern_size)
        if found:
            detected_frames.append((frame_no, frame))
            print("chessboard found", len(detected_frames))

    if not detected_frames:
        print("no chessboard patterns detected in the video.")
        return

    selected_frames = select_uniform_frames(detected_frames, num_frames)
    
    for idx, (frame_no, frame) in enumerate(selected_frames):
        output_path = os.path.join(output_dir, f"frame_{idx}.png")
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"saved frame {frame_no} to {output_path}")

    print(f"total frames detected: {len(detected_frames)}")
    print(f"selected {len(selected_frames)} frames for calibration dataset.")

if __name__ == "__main__":
    root = find_project_root()
    base_folder = f"{root}/images/STEREOS"
    videos_base_path = f"{root}/videos/STEREOS"
    
    cam_1 = f"{videos_base_path}/STEREO_A/1_cam_intrinsic.MOV"
    cam_2 = f"{videos_base_path}/STEREO_A/2_cam_intrinsic.MOV"
    
    cam_3 = f"{videos_base_path}/STEREO_B/3_cam_intrinsic.MOV"
    cam_4 = f"{videos_base_path}/STEREO_B/4_cam_intrinsic.MOV"
    
    videos = [cam_1, cam_2, cam_3, cam_4]
    
    cameras_output_paths = {
        "CAM_1": os.path.join(base_folder, "STEREO_A/CAMERA_1", "intrinsic_frames"),
        "CAM_2": os.path.join(base_folder, "STEREO_A/CAMERA_2", "intrinsic_frames"),
        
        "CAM_3": os.path.join(base_folder, "STEREO_B/CAMERA_3", "intrinsic_frames"),
        "CAM_4": os.path.join(base_folder, "STEREO_B/CAMERA_4", "intrinsic_frames")
    }
    
    # pattern variables
    chessboard_size = (9, 6)
    square_size_mm = 60
    
    i = 0
    for cam_name, folder in cameras_output_paths.items():
        process_video(
            record_name=cam_name,
            video_path=videos[i],
            output_dir=folder,
            pattern_size=chessboard_size,
            num_frames=150,
            skip_seconds=1
        )
        print(f"--------------------------- {cam_name} DONE ---------------------------")
        i += 1
