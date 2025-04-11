import cv2
import numpy as np
import os
from pathlib import Path
import glob
import json
import sys
import shutil

from utils.utils import get_frame_size, print_image


def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")

def save_intrinsics_to_json(intrinsics_dict, output_path):
    with open(output_path, "w") as f:
        json.dump(intrinsics_dict, f, indent=4)

def compute_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    """
    how close our predicted positions of the chessboard corners are to where they actually appear in the picture. 
    if the difference is small, it means our calibration are very accurate
    
    rule = less than 1.0 is good 
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


def filter_images_by_error(objpoints, imgpoints, rvecs, tvecs, K, D, threshold):
    filtered_obj = []
    filtered_img = []
    filtered_indices = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(objpoints[i])
        if error < threshold:
            filtered_obj.append(objpoints[i])
            filtered_img.append(imgpoints[i])
            filtered_indices.append(i)
    return filtered_obj, filtered_img, filtered_indices


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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)

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
    initial_imgs = []

    i = 0
    for cam_img in images:
        img = cv2.imread(cam_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        # if corners are found, add object points, image points (after refining them)
        if ret:
            # store the same object for each image
            objpoints.append(objp)
            # get all the coordinates of the corners
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            # append the corners to the imgpoints list
            imgpoints.append(corners)

            # HERE IS HOW I'M STORING IMAGES
            initial_imgs.append(cam_img)
        # if i < 3:
        #     cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        #     print_image(img)
        #     i += 1
        #     cv2.waitKey(1000)

    # cv2.destroyAllWindows()
    count_before = len(objpoints)
    # original intrinsic parameters
    # ret = reprojection error
    # K = intrinsic matrix
    # D = distortion coefficients -> how much the lens bends the image (q tanto curva la imagen el lente)
    # R = rotation vectors
    # T = translation vectors
    ret, K, D, R, T = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)
    height, width, channels = img.shape
    initial_error = compute_reprojection_error(objpoints, imgpoints, R, T, K, D)

    # NOTE: compute reprojection error and remove outliers from the overall calibration
    filtered_obj, filtered_img, filtered_indices = filter_images_by_error(
        objpoints,
        imgpoints,
        R,
        T,
        K,
        D,
        threshold=0.06, # 0.6 work the best
    )
    count_after = len(filtered_obj)
    if count_after >= 3:
        ret, K, D, R, T = cv2.calibrateCamera(
            filtered_obj, 
            filtered_img, 
            frame_size, 
            None, 
            None
        )
        error = compute_reprojection_error(filtered_obj, filtered_img, R, T, K, D)
    else:
        error = initial_error
        print("Not enough images after filtering; using initial calibration error")

    # NOTE: save filtered images used to achieve the reprojection error - debugging purposes
    filtered_folder = os.path.join(os.path.dirname(folder), "filtered_" + os.path.basename(folder))
    
    # if folder exist, remove the content inside
    if os.path.exists(filtered_folder):
        shutil.rmtree(filtered_folder)
        
    if not os.path.exists(filtered_folder):
        os.makedirs(filtered_folder)
        
    for idx in filtered_indices:
        src = initial_imgs[idx]
        dest = os.path.join(filtered_folder, os.path.basename(src))
        shutil.copy2(src, dest)

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
    return K, D, error, count_before, count_after, NEW_K, roi


def process_intrinsic(base_folder, output_path, chessboard_size, square_size_mm):
    cameras = {
        "CAM_1": os.path.join(base_folder, "STEREO_A/CAMERA_1", "intrinsic_frames"),
        "CAM_2": os.path.join(base_folder, "STEREO_A/CAMERA_2", "intrinsic_frames"),
        "CAM_3": os.path.join(base_folder, "STEREO_B/CAMERA_3", "intrinsic_frames"),
        "CAM_4": os.path.join(base_folder, "STEREO_B/CAMERA_4", "intrinsic_frames")
    }
    intrinsics = {}
    for cam_name, folder in cameras.items():
        K, D, error, count_before, count_after, NEW_K, roi = calibrate_camera_from_folder(
            folder, 
            chessboard_size, 
            square_size_mm
        )

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
            "K_undistort": NEW_K.tolist(),
            "roi_undistort": list(roi),
            "count_bfore": count_before,
            "count_after": count_after
        }

        print(f"{cam_name} calibrated: {count_after} images, error: {error:.4f}")
    save_intrinsics_to_json(intrinsics, output_path)
    print("Intrinsic parameters saved to", output_path)


if __name__ == "__main__":
    root = find_project_root()
    base_path = f"{root}/images/STEREOS"
    output_path = f"{root}/output/V2_intrinsic_params.json"
    chessboard_size = (9, 6)
    square_size_mm = 60
    process_intrinsic(base_path, output_path, chessboard_size, square_size_mm)
