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

def upscale_image(image, scale_factor=2):
    # makes the image larger to reveal more details
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def detect_chessboard(frame, pattern_size, upscale_factor=2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    # TODO: 1. try automatic detection on the full image
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners

    # TODO: 2. try upscaling the image
    upscaled = upscale_image(gray, scale_factor=upscale_factor)
    cv2.imshow("UPSCALED", upscaled) 
    cv2.waitKey(1) 
    ret, corners = cv2.findChessboardCorners(upscaled, pattern_size, flags)
    if ret:
        # adjust coordinates back to the original scale
        corners = corners / upscale_factor
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners

    return None, None

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
        found, corners = detect_chessboard(frame, pattern_size)
        if found:
            detected_frames.append((frame_no, frame))
            print("chessboard found", len(detected_frames))

    if not detected_frames:
        print("no chessboard patterns detected in the video.")
        return

    selected_frames = select_uniform_frames(detected_frames, num_frames)
    
    for idx, (frame_no, frame) in enumerate(selected_frames):
        output_path = os.path.join(output_dir, f"frame_{record_name}_{frame_no:04d}.png")
        cv2.imwrite(output_path, frame)
        print(f"saved frame {frame_no} to {output_path}")

    print(f"total frames detected: {len(detected_frames)}")
    print(f"selected {len(selected_frames)} frames for calibration dataset.")

if __name__ == "__main__":
    # input
    camera = "CAM_B_RIGHT"
    record_name = "cam_unique_view"
    root = find_project_root()
    video_path = f"{root}/videos/{record_name}.MOV"

    # output
    output_path = f"{root}/images/cameras/stereo-left/"

    # pattern variables
    chessboard_size = (9, 6)
    square_size_mm = 60

    process_video(
        record_name=record_name,
        video_path=video_path,
        output_dir=output_path,
        pattern_size=chessboard_size,
        num_frames=50,
        skip_seconds=1
    )
