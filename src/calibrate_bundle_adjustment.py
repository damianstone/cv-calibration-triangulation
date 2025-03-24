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


def perform_bundle_adjustment(calib_groups_folder, intrinsics, chessboard_size, square_size_mm):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    camera_ids = ["LEFT_CAM_A", "LEFT_CAM_B", "RIGHT_CAM_A", "RIGHT_CAM_B"]
    image_points = {cam: [] for cam in camera_ids}
    object_points = []

    group_folders = sorted(glob.glob(os.path.join(calib_groups_folder, "*")))
    for group in tqdm(group_folders, desc="Processing calibration groups"):
        group_found = True
        group_corners = {}
        # TODO: change this
        # For each camera, try to load the chessboard image and detect corners
        for cam in camera_ids:
            img_path = os.path.join(group, cam.lower() + ".png")
            if not os.path.exists(img_path):
                group_found = False
                break
            img = cv2.imread(img_path)
            ret, corners = detect_chessboard(img, chessboard_size)
            if not ret:
                group_found = False
                break
            group_corners[cam] = corners
        if group_found:
            # Save the detected 2D points for each camera and common 3D points
            for cam in camera_ids:
                image_points[cam].append(group_corners[cam])
            object_points.append(objp)

    # Now run calibration (bundle adjustment) for each camera to optimize its pose
    camera_poses = {}
    for cam in camera_ids:
        if len(image_points[cam]) == 0:
            continue
        # Use image size from the last loaded image (assume all images are same size)
        img_size = (img.shape[1], img.shape[0])
        # Get initial guess of camera intrinsics from provided data
        K_initial = np.array(intrinsics[cam]["K"])
        D_initial = np.array(intrinsics[cam]["D"])
        # Calibrate camera using all calibration images (this optimizes the camera poses)
        ret, cam_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points[cam], img_size, K_initial, D_initial,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        # For a demo, average all rotation and translation vectors from different images
        avg_rvec = np.mean(np.array(rvecs), axis=0)
        avg_tvec = np.mean(np.array(tvecs), axis=0)
        camera_poses[cam] = {
            "K": cam_matrix.tolist(),                     # Optimized camera matrix
            "dist_coeffs": dist_coeffs.tolist(),            # Optimized distortion coefficients
            "avg_rvec": avg_rvec.flatten().tolist(),        # Average rotation (camera orientation)
            "avg_tvec": avg_tvec.flatten().tolist(),        # Average translation (camera position)
            "num_images": len(rvecs)
        }
    return camera_poses


if __name__ == "__main__":
    # Find the project root and define the folder with calibration images
    root = find_project_root()
    calib_groups_folder = os.path.join(root, "images", "bundle_adjustment", "GROUPS")

    # Load intrinsic parameters from file
    intrinsic_path = os.path.join(root, "output", "intrinsic_params.json")
    with open(intrinsic_path, "r") as f:
        intrinsics = json.load(f)

    chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column
    square_size_mm = 60       # Size of one chessboard square in millimeters

    # Run bundle adjustment to compute all camera poses in a shared coordinate system
    bundle_adjusted_poses = perform_bundle_adjustment(
        calib_groups_folder, intrinsics, chessboard_size, square_size_mm)

    # Save the optimized poses to a JSON file for later use (e.g., triangulation)
    output_path = os.path.join(root, "output", "bundle_adjusted_poses.json")
    save_json(bundle_adjusted_poses, output_path)
    print("Bundle adjusted poses saved to", output_path)
