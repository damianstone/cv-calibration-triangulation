import cv2
import numpy as np
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm


def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(f"Project root marker '{marker}' not found.")


def save_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def detect_chessboard(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return ret, corners


def perform_bundle_adjustment(groups_folder, intrinsics, chessboard_size, square_size_mm):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    group_folders = sorted(glob.glob(os.path.join(groups_folder, "*")))
    print(group_folders)


    for group in tqdm(group_folders, desc="Processing groups"):
        print("GROUP -> ", group)
        # photos taken of the chessboard at the same time
        img_left_a = os.path.join(group, "left_a.png")
        img_left_b = os.path.join(group, "left_b.png")
        img_right_a = os.path.join(group, "right_a.png")
        img_right_b = os.path.join(group, "right_b.png")

        if not (os.path.exists(img_left_a) and os.path.exists(img_left_b) and
                os.path.exists(img_right_a) and os.path.exists(img_right_b)):
            continue

        im_left_a = cv2.imread(img_left_a)
        im_left_b = cv2.imread(img_left_b)
        im_right_a = cv2.imread(img_right_a)
        im_right_b = cv2.imread(img_right_b)


if __name__ == "__main__":
    root = find_project_root()
    groups_folder = os.path.join(root, "images", "full-stereo", "GROUPS")

    intrinsic_path = os.path.join(root, "output", "intrinsic_params.json")
    with open(intrinsic_path, "r") as f:
        intrinsics = json.load(f)

    chessboard_size = (9, 6)
    square_size_mm = 60

    # Run bundle adjustment to compute all camera poses in a shared coordinate system
    bundle_adjusted_poses = perform_bundle_adjustment(
        groups_folder, intrinsics, chessboard_size, square_size_mm)

    # Save the optimized poses to a JSON file for later use (e.g., triangulation)
    output_path = os.path.join(root, "output", "bundle_adjusted_poses.json")
    save_json(bundle_adjusted_poses, output_path)
    print("Bundle adjusted poses saved to", output_path)
