import cv2
import av
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
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return ret, corners


def select_uniform_frames(frames_list, n):
    if len(frames_list) <= n:
        return frames_list
    indices = np.linspace(0, len(frames_list) - 1, n, dtype=int)
    return [frames_list[i] for i in indices]


def process_stereo_videos(
        cam_1,
        cam_2,
        output_dir,
        pattern_size,
        num_frames,
        skip_seconds=1):
    # create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # open both video files
    cam_1_cap = cv2.VideoCapture(cam_1)
    cam_2_cap = cv2.VideoCapture(cam_2)
    
    cam_1_fps = cam_1_cap.get(cv2.CAP_PROP_FPS)
    cam_2_fps = cam_2_cap.get(cv2.CAP_PROP_FPS)
    
    cam_1_total_frames = int(cam_1_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cam_2_total_frames = int(cam_2_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cam_1_duration = int(cam_1_total_frames / cam_1_fps)
    cam_2_duration = int(cam_2_total_frames / cam_2_fps)
    
    duration_seconds = min(cam_1_duration, cam_2_duration)

    detected_pairs = []

    # iterate through time, reading a frame from each video at the same timestamp
    for sec in np.arange(0, duration_seconds):
        if len(detected_pairs) == 5:
            break
        cam_1_frame_no = int(sec * cam_1_fps)
        cam_2_frame_no = int(sec * cam_2_fps)
        
        cam_1_cap.set(cv2.CAP_PROP_POS_FRAMES, cam_1_frame_no)
        cam_2_cap.set(cv2.CAP_PROP_POS_FRAMES, cam_2_frame_no)
        
        ret_cam_1, cam_1_frame = cam_1_cap.read()
        ret_cam_2, cam_2_frame = cam_2_cap.read()

        if not ret_cam_1 or not ret_cam_2:
            break

        # detect chessboard in both frames
        found_cam_1, _ = detect_chessboard(cam_1_frame, pattern_size)
        found_cam_2, _ = detect_chessboard(cam_2_frame, pattern_size)
        if found_cam_1 and found_cam_2:
            detected_pairs.append((cam_1_frame_no, cam_1_frame, cam_2_frame_no, cam_2_frame))
            print("Synchronized chessboard pair found:", len(detected_pairs))

    cam_1_cap.release()
    cam_2_cap.release()

    if not detected_pairs:
        print("No synchronized chessboard pairs found.")
        return

    # uniformly select pairs if more than desired
    selected_pairs = select_uniform_frames(detected_pairs, num_frames)

    # save each pair in its own folder with left and right images
    for idx, (cam_1_frame_no, cam_1_frame, cam_2_frame_no, cam_2_frame) in enumerate(selected_pairs):
        pair_folder = os.path.join(output_dir, f"{idx:03d}")
        os.makedirs(pair_folder, exist_ok=True)
        
        cam_1_path = os.path.join(pair_folder, f"cam_1.png")
        cam_2_path = os.path.join(pair_folder, f"cam_2.png")
        
        cv2.imwrite(cam_1_path, cam_1_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(cam_2_path, cam_2_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        print(f"Saved pair {idx}: left frame {cam_1_frame_no}, right frame {cam_2_frame_no}")

    print("Total pairs detected:", len(detected_pairs))
    print("Selected pairs for calibration:", len(selected_pairs))


if __name__ == "__main__":
    root = find_project_root()
    base_folder = f"{root}/images/STEREOS"
    videos_base_path = f"{root}/videos/STEREOS"
    
    cam_1 = f"{videos_base_path}/STEREO_A/1_cam_extrinsic.mp4"
    cam_2 = f"{videos_base_path}/STEREO_A/2_cam_extrinsic.mp4"
    
    cam_3 = f"{videos_base_path}/STEREO_B/3_cam_extrinsic.mp4"
    cam_4 = f"{videos_base_path}/STEREO_B/4_cam_extrinsic.mp4"

    stereos = [(cam_1, cam_2), (cam_3, cam_4)]

    cameras_output_paths = {
        "STEREO_A": os.path.join(base_folder, "STEREO_A", "stereo_frames"),
        "STEREO_B": os.path.join(base_folder, "STEREO_B", "stereo_frames"),
    }

    # pattern variables
    chessboard_size = (9, 6)
    square_size_mm = 60

    for i, (cam_name, folder) in enumerate(cameras_output_paths.items()):
        stereo = stereos[i]
        process_stereo_videos(
            cam_1=stereo[0],
            cam_2=stereo[1],
            output_dir=folder,
            pattern_size=chessboard_size,
            num_frames=200,
            skip_seconds=0.5
        )
        print(f"--------------------------- {cam_name} DONE ---------------------------")

# remover 53 al final 4_cam