import cv2
import numpy as np
import os
import glob
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from utils.utils import print_image

"""
How to improve extrinsic
- get images again in better quality X
- pre-process each image for better chessboard detection X
- djust the corner refinement parameters -> WORKS
        window size defines the area around an initially detected corner that the 
        algorithm examines to refine its position
- Fine-tuning detection parameters WORKS
"""


def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")


def save_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def save_filtered_pairs(filtered_pairs, folder):
    filtered_folder = os.path.join(os.path.dirname(
        folder), "filtered_" + os.path.basename(folder))

    if not os.path.exists(filtered_folder):
        os.makedirs(filtered_folder)

    for idx, (left_path, right_path) in enumerate(filtered_pairs):

        pair_folder = os.path.join(filtered_folder, f"{idx:03d}")
        os.makedirs(pair_folder, exist_ok=True)

        left_dest = os.path.join(pair_folder, "left.png")
        right_dest = os.path.join(pair_folder, "right.png")

        shutil.copy2(left_path, left_dest)
        shutil.copy2(right_path, right_dest)

    print("Filtered pairs saved to", filtered_folder)


def check_frame_size(img, expected_size=None):
    current_size = img.shape[:2]
    if expected_size is None:
        return current_size
    if current_size != expected_size:
        print(f"Error: Image size {current_size} does not match expected {expected_size}")
    return expected_size

def detect_chessboard(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # use adaptive threshold and normalization flags for better detection
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        # stricter termination criteria for improved subpixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    return ret, corners

def compute_reprojection_error_pair(objp, corners, K, D):
    # Estimate pose for the single image
    ret, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    error = cv2.norm(corners, proj, cv2.NORM_L2) / len(objp)
    return error


def calibrate_stereo_from_folder(
        folder,
        chessboard_size,
        square_size_mm,
        intrinsic_left,
        intrinsic_right,
        error_threshold=0.05):

    subfolders = sorted(glob.glob(os.path.join(folder, "*")))
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    filtered_image_paths = []
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    # adjust the corner refinement parameters to use a smaller window and stricter convergence criteria,
    # improving precision in corner localization.
    # This yields more accurate chessboard points and lower calibration errors
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

    # Convert intrinsics to numpy arrays
    K_left = np.array(intrinsic_left["K"])
    D_left = np.array(intrinsic_left["D"])
    K_right = np.array(intrinsic_right["K"])
    D_right = np.array(intrinsic_right["D"])

    expected_size = None
    i = 0
    for subfolder in tqdm(subfolders, desc="Processing folders"):
        left_img_path = os.path.join(subfolder, "left.png")
        right_img_path = os.path.join(subfolder, "right.png")
        if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
            continue

        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        # NOTE: alernt if there is a different frame size
        expected_size = check_frame_size(left_img, expected_size)
        expected_size = check_frame_size(right_img, expected_size)

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        # ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
        ret_left, corners_left = detect_chessboard(left_img, chessboard_size)
        ret_right, corners_right = detect_chessboard(right_img, chessboard_size)
        if ret_left and ret_right:
            corners_left = cv2.cornerSubPix(
                gray_left, corners_left, winSize, zeroZone, criteria)
            corners_right = cv2.cornerSubPix(
                gray_right, corners_right, winSize, zeroZone, criteria)

            # Compute error for each image individually using solvePnP
            error_l = compute_reprojection_error_pair(objp, corners_left, K_left, D_left)
            error_r = compute_reprojection_error_pair(objp, corners_right, K_right, D_right)
            avg_error = (error_l + error_r) / 2.0

            # keep the pair if error is below threshold
            if avg_error < error_threshold:
                objpoints.append(objp)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
                filtered_image_paths.append((left_img_path, right_img_path))
            
            # if i < 5:
            #     cv2.drawChessboardCorners(left_img, chessboard_size, corners_left, ret_left)
            #     cv2.drawChessboardCorners(right_img, chessboard_size, corners_right, ret_right)
            #     print_image(left_img)
            #     print_image(right_img)
            #     i += 1
            #     cv2.waitKey(1000)

    if len(objpoints) < 3:
        raise Exception("Not enough valid pairs for stereo calibration")

    image_size = gray_left.shape[::-1]
    # stereo calibration with filtered pairs
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K_left, D_left, K_right, D_right,
        image_size,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"VALID PAIRS -> {len(objpoints)} REPROJECTION ERROR -> {round(ret, 3)}")

    try:
        save_filtered_pairs(filtered_image_paths, folder)
    except:
        raise "error saving stereo pairs"

    return {
        "rotation_matrix": R.tolist(),
        "translation_vector": T.tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        "reprojection_error": ret,
        "num_valid_pairs": len(objpoints)
    }


if __name__ == "__main__":
    root = find_project_root()

    # Load intrinsic parameters computed previously
    intrinsic_path = os.path.join(root, "output", "intrinsic_params.json")
    with open(intrinsic_path, "r") as f:
        intrinsics = json.load(f)

    chessboard_size = (9, 6)
    square_size_mm = 60

    # Stereo-left: LEFT_CAM_A and LEFT_CAM_B
    stereo_left_folder = os.path.join(root, "images", "cameras", "stereo-left", "V2_STEREO")
    stereo_left_params = calibrate_stereo_from_folder(
        stereo_left_folder,
        chessboard_size,
        square_size_mm,
        intrinsics["LEFT_CAM_A"],
        intrinsics["LEFT_CAM_B"]
    )

    # Stereo-right: RIGHT_CAM_A and RIGHT_CAM_B
    stereo_right_folder = os.path.join(root, "images", "cameras", "stereo-right", "V2_STEREO")
    stereo_right_params = calibrate_stereo_from_folder(
        stereo_right_folder, chessboard_size, square_size_mm,
        intrinsics["RIGHT_CAM_A"], intrinsics["RIGHT_CAM_B"]
    )

    stereo_params = {
        "stereo_left": stereo_left_params,
        "stereo_right": stereo_right_params
    }

    output_path = os.path.join(root, "output", "stereo_params_3.json")
    save_json(stereo_params, output_path)
    print("Stereo calibration parameters saved to", output_path)
