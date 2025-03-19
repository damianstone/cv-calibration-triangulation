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
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return ret, corners


def select_uniform_frames(frames_list, n):
    if len(frames_list) <= n:
        return frames_list
    indices = np.linspace(0, len(frames_list) - 1, n, dtype=int)
    return [frames_list[i] for i in indices]


def process_video(video_path, output_dir, pattern_size, num_frames):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    detected_frames = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        found, corners = detect_chessboard(frame, pattern_size)
        if found:
            detected_frames.append((frame_index, frame))
        frame_index += 1

    cap.release()

    if not detected_frames:
        print("No chessboard patterns detected in the video.")
        return

    selected_frames = select_uniform_frames(detected_frames, num_frames)

    for idx, (frame_no, frame) in enumerate(selected_frames):
        output_path = os.path.join(output_dir, f"frame_{frame_no:04d}.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_no} to {output_path}")

    print(f"Total frames detected: {len(detected_frames)}")
    print(f"Selected {len(selected_frames)} frames for calibration dataset.")


if __name__ == "__main__":
    # TODO: video path
    camera = "CAM_A_LEFT"
    root = find_project_root()
    video_path = f"{root}/{camera}/1_stereo_view.MOV"

    # TODO: output path
    output_path = f"{root}/images/cameras/stereo-left/CAM_A"

    # TODO: pattern variables
    chessboard_size = (10, 7)
    square_size_mm = 60

    # TODO: number of frames to select)
    process_video(
        video_path=video_path,
        output_dir=output_path,
        pattern_size=chessboard_size,
        num_frames=30,
    )
