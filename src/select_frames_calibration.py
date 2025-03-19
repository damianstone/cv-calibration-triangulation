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

def manual_mark_corners(image, pattern_size):
    # lets you click on the corners manually when automatic detection fails
    points = []
    required_points = pattern_size[0] * pattern_size[1]

    def click_event(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("manual marking", image)

    cv2.imshow("manual marking", image)
    cv2.setMouseCallback("manual marking", click_event)
    print(f"please click on {required_points} corners in order. press 'esc' to exit if needed.")
    while len(points) < required_points:
        if cv2.waitKey(1) & 0xFF == 27:  # escape key to exit
            break
    cv2.destroyWindow("manual marking")
    corners = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    return corners

def detect_chessboard(frame, pattern_size, upscale_factor=2, border_fraction=(0.0238, 0.0333)):
    # convert to grayscale
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

    # TODO: 3. crop out the known border (15mm on each side) based on the board dimensions
    # the board is 630x450 mm with a 15mm border → inner board covers ~95.24% (600/630) width and ~93.33% (420/450) height
    # h, w = gray.shape
    # crop_x = int(w * 0.0238)  # about 2.38% from left/right
    # crop_y = int(h * 0.0333)  # about 3.33% from top/bottom
    # cropped = gray[crop_y:h - crop_y, crop_x:w - crop_x]
    # cv2.imshow("CROPPED", cropped)
    # cv2.waitKey(1) 
    # ret, corners = cv2.findChessboardCorners(cropped, pattern_size, flags)
    # if ret:
    #     # adjust corners back to original image coordinates
    #     corners = corners + np.array([[[crop_x, crop_y]]], dtype=np.float32)
    #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #     corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    #     return ret, corners

    return None, None
    # 4. if all else fails, let the user mark the corners manually
    # print("automatic detection failed, please mark corners manually.")
    # corners = manual_mark_corners(frame.copy(), pattern_size)
    # ret = (corners.shape[0] == pattern_size[0] * pattern_size[1])
    # return ret, corners


def select_uniform_frames(frames_list, n):
    """
    selects a fixed number of frames evenly from the list of good frames found
    """
    if len(frames_list) <= n:
        return frames_list
    indices = np.linspace(0, len(frames_list) - 1, n, dtype=int)
    return [frames_list[i] for i in indices]


def process_video(video_path, output_dir, pattern_size, num_frames, skip_seconds=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # get frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = int(total_frames / fps)  # total duration in seconds

    detected_frames = []
    
    # for every second in the video (skipping seconds as specified)
    for sec in range(0, duration_seconds, skip_seconds):
        frame_no = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # jump directly to the desired frame
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
        output_path = os.path.join(output_dir, f"frame_{frame_no:04d}.png")
        cv2.imwrite(output_path, frame)
        print(f"saved frame {frame_no} to {output_path}")

    print(f"total frames detected: {len(detected_frames)}")
    print(f"selected {len(selected_frames)} frames for calibration dataset.")

if __name__ == "__main__":
    # TODO: video path
    camera = "CAM_A_LEFT"
    root = find_project_root()
    video_path = f"{root}/videos/{camera}/cam_unique_view.MOV"

    # TODO: output path
    output_path = f"{root}/images/cameras/stereo-left/CAM_A"

    # TODO: pattern variables
    chessboard_size = (10, 7)
    square_size_mm = 60
    border_mm = 15

    # TODO: number of frames to select)
    process_video(
        video_path=video_path,
        output_dir=output_path,
        pattern_size=chessboard_size,
        num_frames=30,
        skip_seconds=1
    )
