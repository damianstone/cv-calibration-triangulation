import cv2
import numpy as np
import os
from pathlib import Path
import glob
import json

def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(f"Project root marker '{marker}' not found starting from {current}")

def compute_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    Compute the average reprojection error over all images.
    """
    total_error = 0
    total_points = 0
    for i in range(len(object_points)):
        projected_imgpoints, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], projected_imgpoints, cv2.NORM_L2)
        total_error += error**2
        total_points += len(object_points[i])
    return np.sqrt(total_error / total_points)

def calibrate_camera_from_folder(folder, chessboard_size, square_size_mm):
    """
    Go over all images in a folder, detect chessboard corners, and run camera calibration.
    Returns the camera matrix, distortion coefficients, reprojection error, and image count.
    """
    # Create object points: a grid of (x,y,0) coordinates multiplied by the square size.
    objp = np.zeros((chessboard_size[1]*chessboard_size[0], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= square_size_mm

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    # Look for PNG or JPG images.
    images = glob.glob(os.path.join(folder, "*.png"))
    if not images:
        images = glob.glob(os.path.join(folder, "*.jpg"))
    if not images:
        print(f"No images found in {folder}")
        return None, None, None, 0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            # Refine corner locations for higher accuracy.
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
    if not objpoints:
        print(f"No chessboard corners detected in {folder}")
        return None, None, None, 0

    ret_val, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    error = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
    return camera_matrix, dist_coeffs, error, len(objpoints)

def save_intrinsics_to_json(intrinsics_dict, output_path):
    with open(output_path, "w") as f:
        json.dump(intrinsics_dict, f, indent=4)

def process_intrinsic(base_folder, output_path, chessboard_size, square_size_mm):
    """
    Process the calibration dataset for each camera and save the intrinsic parameters.
    """
    # Define the folders for each camera dataset.
    cameras = {
        "LEFT_CAM_A": os.path.join(base_folder, "stereo-left", "CAM_A"),
        "LEFT_CAM_B": os.path.join(base_folder, "stereo-left", "CAM_B"),
        "RIGHT_CAM_A": os.path.join(base_folder, "stereo-right", "CAM_A"),
        "RIGHT_CAM_B": os.path.join(base_folder, "stereo-right", "CAM_B")
    }
    intrinsics = {}
    for cam_name, folder in cameras.items():
        print(f"Calibrating {cam_name} using images from {folder}")
        mtx, dist, error, count = calibrate_camera_from_folder(folder, chessboard_size, square_size_mm)
        if mtx is None:
            print(f"Calibration failed for {cam_name}.")
            continue
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]
        intrinsics[cam_name] = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "D": dist.flatten().tolist(),
            "K": mtx.tolist(),
            "reprojection_error": float(error),
            "num_images": count
        }
        print(f"{cam_name} calibrated: {count} images, error: {error:.4f}")
    save_intrinsics_to_json(intrinsics, output_path)
    print("Intrinsic parameters saved to", output_path)

if __name__ == "__main__":
    root = find_project_root()
    base_path = f"{root}/images/cameras"
    output_path = f"{root}/data/intrinsic_params.json"
    chessboard_size = (9, 6)
    square_size_mm = 60
    process_intrinsic(base_path, output_path, chessboard_size, square_size_mm)


    """
      "LEFT_CAM_A": {
          "fx": 1800.0,
          "fy": 1805.0,
          "cx": 1920.0,
          "cy": 1080.0,
          "D": [-0.25, 0.05, 0.0, 0.0, -0.02],
          "K": [
            [1800.0, 0, 1920.0],
            [0, 1805.0, 1080.0],
            [0, 0, 1]
          ]
        },
      "LEFT_CAM_B": {
          "fx": 1800.0,
          "fy": 1805.0,
          "cx": 1920.0,
          "cy": 1080.0,
          "D": [-0.25, 0.05, 0.0, 0.0, -0.02],
          "K": [
            [1800.0, 0, 1920.0],
            [0, 1805.0, 1080.0],
            [0, 0, 1]
          ]
        },
      "RIGHT_CAM_A": {
          "fx": 1800.0,
          "fy": 1805.0,
          "cx": 1920.0,
          "cy": 1080.0,
          "D": [-0.25, 0.05, 0.0, 0.0, -0.02],
          "K": [
            [1800.0, 0, 1920.0],
            [0, 1805.0, 1080.0],
            [0, 0, 1]
          ]
        },
      "RIGHT_CAM_B": {
          "fx": 1800.0,
          "fy": 1805.0,
          "cx": 1920.0,
          "cy": 1080.0,
          "D": [-0.25, 0.05, 0.0, 0.0, -0.02],
          "K": [
            [1800.0, 0, 1920.0],
            [0, 1805.0, 1080.0],
            [0, 0, 1]
          ]
        }
    """