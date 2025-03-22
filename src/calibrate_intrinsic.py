import cv2
import numpy as np
import os
from pathlib import Path
import glob
import json
import sys

from utils.utils import get_frame_size, print_image

def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")

def compute_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    average error between where the calibration model predicts the chessboard corners 
    should appear (after projecting 3D points into 2D) and where they were actually detected. 
    A lower error means a more accurate calibration.
    """
    total_error = 0
    total_points = 0
    for i in range(len(object_points)):
        projected_imgpoints, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], projected_imgpoints, cv2.NORM_L2)
        total_error += error**2
        total_points += len(object_points[i])
    return np.sqrt(total_error / total_points)


def calibrate_camera_from_folder(folder, chessboard_size, square_size_mm):
    """
    Go over all images in a folder, detect chessboard corners, and run camera calibration.
    Returns the camera matrix, distortion coefficients, reprojection error, and image count.
    """
    images = glob.glob(os.path.join(folder, "*.png"))
    if not images:
        images = glob.glob(os.path.join(folder, "*.jpg"))
    if not images:
        print(f"No images found in {folder}")
        return None, None, None, 0

    frame_size = get_frame_size(images[0])

    # how exact we want to be find the corners of each square in the chessboard
    # 30 = max number of iterations
    # 0.001 = stop if the change in the corner position is less than 0.001 - convergence threshold
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # create 3D points of the chessboard corners
    # each row represents a coordinate of a corner of the chessboard
    # corner = interesection between two squares
    objp = np.zeros(
        (chessboard_size[0] * chessboard_size[1], 3),
        np.float32
    )

    # it generates a grid of (x, y) coordinates for chessboard corners and stores them as 3D points with z = 0 for calibration
    # z = 0 because the chessboard is flat
    objp[:, :2] = np.mgrid[
        0:chessboard_size[0],
        0:chessboard_size[1]
    ].T.reshape(-1, 2)

    objp = objp * square_size_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    i = 0
    for cam_img in images:
        img = cv2.imread(cam_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # TODO: make a way to don't count frames that have high reprojection error, check if the imahe
        # have more than 1 reprojection error, if its less then count it for the general calibration if not just skip

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # if corners are found, add object points, image points (after refining them)
        if ret:
            # store the same object for each image
            objpoints.append(objp)

            # get all the coordinates of the corners
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # append the corners to the imgpoints list
            imgpoints.append(corners)

        # if i < 3:
        #     cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        #     print_image(img)
        #     i += 1
        #     cv2.waitKey(1000)

    # cv2.destroyAllWindows()

    # original intrinsic parameters
    # ret = reprojection error
    # K = intrinsic matrix
    # D = distortion coefficients -> how much the lens bends the image (q tanto curva la imagen el lente)
    # R = rotation vectors
    # T = translation vectors
    ret, K, D, R, T = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        frame_size,
        None,
        None
    )
    height, width, channels = img.shape

    # NOTE: this is not necessary for the intrinsics or calibration in general, save them just in case
    # new camera matrix for undistortion, useful to test the original intrinsic parameters
    # new_k only used to undistort images after calibration, to correct distorsiobned images
    # roi =  region of interest, area of the undistorted image where the correction is valid
    NEW_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        D,
        (width, height),
        1,
        (width, height)
    )

    # lower the best
    # how precise is the calibration
    error = compute_reprojection_error(
        objpoints,
        imgpoints,
        R,
        T,
        K,
        D
    )

    return K, D, error, len(objpoints), NEW_K, roi


def save_intrinsics_to_json(intrinsics_dict, output_path):
    with open(output_path, "w") as f:
        json.dump(intrinsics_dict, f, indent=4)


def process_intrinsic(base_folder, output_path, chessboard_size, square_size_mm):
    cameras = {
        "LEFT_CAM_A": os.path.join(base_folder, "stereo-left", "V2_CAM_A"),
        "LEFT_CAM_B": os.path.join(base_folder, "stereo-left", "V2_CAM_B"),
        "RIGHT_CAM_A": os.path.join(base_folder, "stereo-right", "V2_CAM_A"),
        "RIGHT_CAM_B": os.path.join(base_folder, "stereo-right", "V2_CAM_B")
    }
    intrinsics = {}
    for cam_name, folder in cameras.items():
        K, D, error, count, NEW_K, roi = calibrate_camera_from_folder(
            folder, chessboard_size, square_size_mm)

        if K is None:
            raise f"Calibration failed for {cam_name}"

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        intrinsics[cam_name] = {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "D": D.flatten().tolist(),
            "K": K.tolist(),
            "reprojection_error": round(float(error), 3),
            "num_images": count,
            "K_undistort": NEW_K.tolist(),
            "roi_undistort": list(roi)
        }

        print(f"{cam_name} calibrated: {count} images, error: {error:.4f}")
    save_intrinsics_to_json(intrinsics, output_path)
    print("Intrinsic parameters saved to", output_path)


if __name__ == "__main__":
    root = find_project_root()
    base_path = f"{root}/images/cameras"
    output_path = f"{root}/output/intrinsic_params_2.json"
    chessboard_size = (9, 6)
    square_size_mm = 60
    process_intrinsic(base_path, output_path, chessboard_size, square_size_mm)
