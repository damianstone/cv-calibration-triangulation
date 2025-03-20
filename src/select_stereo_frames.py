import cv2
import numpy as np
import os
from pathlib import Path

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

def detect_chessboard(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

def select_uniform_frames(frames_list, n):
    if len(frames_list) <= n:
        return frames_list
    indices = np.linspace(0, len(frames_list) - 1, n, dtype=int)
    return [frames_list[i] for i in indices]

def process_stereo_videos(
    record_name, 
    left_video_path, 
    right_video_path, 
    output_dir, 
    pattern_size, 
    num_frames, 
    skip_seconds=1):
    # create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # open both video files
    left_cap = cv2.VideoCapture(left_video_path)
    right_cap = cv2.VideoCapture(right_video_path)
    left_fps = left_cap.get(cv2.CAP_PROP_FPS)
    right_fps = right_cap.get(cv2.CAP_PROP_FPS)
    left_total_frames = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_total_frames = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    left_duration = int(left_total_frames / left_fps)
    right_duration = int(right_total_frames / right_fps)
    duration_seconds = min(left_duration, right_duration)
    
    detected_pairs = []
    
    # iterate through time, reading a frame from each video at the same timestamp
    for sec in range(0, duration_seconds):
        # if len(detected_pairs) == 5:
        #     break
        left_frame_no = int(sec * left_fps)
        right_frame_no = int(sec * right_fps)
        left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_frame_no)
        right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_frame_no)
        ret_left, left_frame = left_cap.read()
        ret_right, right_frame = right_cap.read()
        if not ret_left or not ret_right:
            break
        
        # detect chessboard in both frames
        found_left, _ = detect_chessboard(left_frame, pattern_size)
        found_right, _ = detect_chessboard(right_frame, pattern_size)
        if found_left and found_right:
            detected_pairs.append((left_frame_no, left_frame, right_frame_no, right_frame))
            print("Synchronized chessboard pair found:", len(detected_pairs))
    
    left_cap.release()
    right_cap.release()
    
    if not detected_pairs:
        print("No synchronized chessboard pairs found.")
        return

    # uniformly select pairs if more than desired
    selected_pairs = select_uniform_frames(detected_pairs, num_frames)
    
    # save each pair in its own folder with left and right images
    for idx, (left_frame_no, left_frame, right_frame_no, right_frame) in enumerate(selected_pairs):
        pair_folder = os.path.join(output_dir, f"{idx:03d}")
        os.makedirs(pair_folder, exist_ok=True)
        left_path = os.path.join(pair_folder, f"left.png")
        right_path = os.path.join(pair_folder, f"right.png")
        cv2.imwrite(left_path, left_frame)
        cv2.imwrite(right_path, right_frame)
        print(f"Saved pair {idx}: left frame {left_frame_no}, right frame {right_frame_no}")
    
    print("Total pairs detected:", len(detected_pairs))
    print("Selected pairs for calibration:", len(selected_pairs))

if __name__ == "__main__":
    root = find_project_root()
    # cam 1
    camera_a = "CAM_A_RIGHT"
    record_name = "1_stereo_view"
    left_video_path = f"{root}/videos/{camera_a}/{record_name}.MOV"
    
    # cam 2
    camera_b = "CAM_B_RIGHT"
    record_name = "1_stereo_view"
    right_video_path = f"{root}/videos/{camera_b}/{record_name}.MOV"

    # output
    output_path = f"{root}/images/cameras/stereo-right/STEREO"

    # pattern variables
    chessboard_size = (9, 6)
    square_size_mm = 60

    process_stereo_videos(
        left_video_path=left_video_path,
        right_video_path=right_video_path,
        record_name=record_name,
        output_dir=output_path,
        pattern_size=chessboard_size,
        num_frames=100,
        skip_seconds=1
    )
