import cv2
import numpy as np
import os
import glob
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from utils.utils import print_image


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

    # if folder exist, remove the content inside
    if os.path.exists(filtered_folder):
        shutil.rmtree(filtered_folder)

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


def save_stereo_maps(
        cam_left_name,
        cam_right_name,
        stereo_map_left,
        stereo_map_right,
        proj_matrix_left,
        proj_matrix_right,
        Q, roi_l, roi_r):
    
    if cam_left_name == "CAM_1" and cam_right_name == "CAM_2":
        stereo_name = "STEREO_A"
    else:
        stereo_name = "STEREO_B"
        
    root = find_project_root()
    # Create a FileStorage object to write data into an XML file
    cv_file = cv2.FileStorage(
        f'{root}/output/{stereo_name}_rectification_params.xml', cv2.FILE_STORAGE_WRITE)

    # Write the stereo map components (for both left and right)
    cv_file.write(f"{cam_left_name}_map_x", stereo_map_left[0])
    cv_file.write(f"{cam_left_name}_map_y", stereo_map_left[1])
    
    cv_file.write(f"{cam_right_name}_map_x", stereo_map_right[0])
    cv_file.write(f"{cam_right_name}_map_y", stereo_map_right[1])

    # Write projection matrices
    cv_file.write(f"{cam_left_name}_projection_matrix", proj_matrix_left)
    cv_file.write(f"{cam_right_name}_projection_matrix", proj_matrix_right)

    # Write disparity-to-depth matrix
    cv_file.write(f"disparity_to_depth_matrix", Q)

    # Write region of interest (ROI) for both cameras
    cv_file.write(f"{cam_left_name}_roi", roi_l)
    cv_file.write(f"{cam_right_name}_roi", roi_r)

    # Save the file
    cv_file.release()
    print("Map parameters saved successfully!")


def check_frame_size(img, expected_size=None):
    current_size = img.shape[:2]
    if expected_size is None:
        return current_size
    if current_size != expected_size:
        print(f"Error: Image size {current_size} does not match expected {expected_size}")
    return expected_size


def detect_chessboard(frame, pattern_size, winSize, criteria):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # use adaptive threshold and normalization flags for better detection
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, winSize, (-1, -1), criteria)

    return ret, corners


def compute_reprojection_error_pair(objp, corners, K, D):
    ret, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    error = cv2.norm(corners, proj, cv2.NORM_L2) / len(objp)
    return error


def stereo_calibration(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    K_left,
    D_left,
    K_right,
    D_right,
    image_size,
):
    """
    Use the original K and D parameters to calibrate the stereo pair (intrinsics)
    """
    # flags control how the calibration works
    flags = 0
    # CALIB_FIX_INTRINSIC -> not change the intrinsic parameters we already calculated
    # meaning we only want to calibrate how the cameras relate to each other not recalibrate them individually
    flags |= cv2.CALIB_FIX_INTRINSIC

    # iteration stopping criteria
    # 30 iterations or until the change in the parameters is less than 0.001
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # stereoCalibrate
    # R = rotation matrix
    # T = translation vector
    # E = essential matrix
    # F = fundamental matrix
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K_left, D_left, K_right, D_right,
        image_size,
        criteria=criteria_stereo,
        flags=flags
    )
    return ret, R, T, E, F


def stereo_rectification(
    K_left_undistort,
    D_left,
    K_right_undistort,
    D_right,
    image_size,
    R,
    T,
):
    """
    Use the undistorted K and D parameters to rectify the stereo pair (intrinsics)
    Rectification alignment step so both cameras share the same coordinate system
    Warps images so they behave as if cameras were perfectly parallel (same y-coordinates for corresponding points)
    Corresponding points are the same row in the rectified images
    """
    rectify_scale = 1
    # rect_l = rectification transform (rotation matrix)
    # rect_r = rectification transform (rotation matrix)
    # proj_matrix_l = projection matrix in new (rectified) coord system for the left camera
    # proj_matrix_r = projection matrix in new (rectified) coord system for the right camera
    # Q = disparity-to-depth mapping matrix
    # roi_l = region of interest in the rectified image
    # roi_r = region of interest in the rectified image
    rect_l, rect_r, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r = cv2.stereoRectify(
        K_left_undistort, D_left, K_right_undistort, D_right,
        image_size,
        R,
        T,
        rectify_scale,
        newImageSize=(0, 0)
    )

    # correct distortion and apply rectification
    # output maps which are pre-made instructions to quickly adjust the images before depth estimation
    stereo_map_left = cv2.initUndistortRectifyMap(
        K_left_undistort, D_left,
        rect_l,
        proj_matrix_l,
        image_size,
        m1type=cv2.CV_32FC1
    )

    stereo_map_right = cv2.initUndistortRectifyMap(
        K_right_undistort, D_right,
        rect_r,
        proj_matrix_r,
        image_size,
        m1type=cv2.CV_32FC1
    )

    return stereo_map_left, stereo_map_right, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r


def calibrate_stereo_from_folder(
        cam_left_name,
        cam_right_name,
        folder,
        chessboard_size,
        square_size_mm,
        intrinsic_left,
        intrinsic_right,
        error_threshold=0.1):
    
    # extrinsics parameters for stereo calibration
    K_left = np.array(intrinsic_left["K"])
    K_left_undistort = np.array(intrinsic_left["K_undistort"])
    D_left = np.array(intrinsic_left["D"])

    K_right = np.array(intrinsic_right["K"])
    K_right_undistort = np.array(intrinsic_right["K_undistort"])
    D_right = np.array(intrinsic_right["D"])

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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    expected_size = None
    i = 0
    for subfolder in tqdm(subfolders, desc="Processing folders"):
        left_img_path = os.path.join(subfolder, "cam_1.png")
        right_img_path = os.path.join(subfolder, "cam_2.png")
        if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
            continue

        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)

        # NOTE: alernt if there is a different frame size
        expected_size = check_frame_size(left_img, expected_size)
        expected_size = check_frame_size(right_img, expected_size)

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = detect_chessboard(
            left_img, chessboard_size, winSize, criteria)
        ret_right, corners_right = detect_chessboard(
            right_img, chessboard_size, winSize, criteria)

        if ret_left and ret_right:
            # compute error for each image individually using solvePnP
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
    # stereo calibration to get the rotation matrix and translation vector
    ret, R, T, E, F = stereo_calibration(
        objpoints, imgpoints_left, imgpoints_right,
        K_left, D_left, K_right, D_right,
        image_size,
    )
    
    print(f"VALID PAIRS -> {len(objpoints)} REPROJECTION ERROR -> {round(ret, 3)}")

    # stereo rectification to align the cameras
    stereo_map_left, stereo_map_right, proj_matrix_left, proj_matrix_right, Q, roi_l, roi_r = stereo_rectification(
        K_left_undistort, D_left, K_right_undistort, D_right,
        image_size,
        R,
        T,
    )
    
    try: 
        save_stereo_maps(
            cam_left_name,
            cam_right_name,
            stereo_map_left,
            stereo_map_right,
            proj_matrix_left,
            proj_matrix_right,
            Q, roi_l, roi_r)
    except:
        raise "error saving stereo maps"
    
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
    base_path = f"{root}/images/STEREOS"
    # Load intrinsic parameters computed previously
    intrinsic_path = os.path.join(root, "output", "2_intrinsic_params.json")
    with open(intrinsic_path, "r") as f:
        intrinsics = json.load(f)

    chessboard_size = (9, 6)
    square_size_mm = 60

    # STEREO A
    # stereo_a_folder = os.path.join(base_path, "STEREO_A", "stereo_frames")
    # stereo_a_params = calibrate_stereo_from_folder(
    #     cam_left_name="CAM_1",
    #     cam_right_name="CAM_2",
    #     folder=stereo_a_folder,
    #     chessboard_size=chessboard_size,
    #     square_size_mm=square_size_mm,
    #     intrinsic_left=intrinsics["CAM_1"],
    #     intrinsic_right=intrinsics["CAM_2"]
    # )

    # STEREO B
    stereo_b_folder = os.path.join(base_path, "STEREO_B", "stereo_frames")
    stereo_b_params = calibrate_stereo_from_folder(
        cam_left_name="CAM_3",
        cam_right_name="CAM_4",
        folder=stereo_b_folder,
        chessboard_size=chessboard_size,
        square_size_mm=square_size_mm,
        intrinsic_left=intrinsics["CAM_3"],
        intrinsic_right=intrinsics["CAM_4"]
    )

    stereo_params = {
        # "STEREO_A": stereo_a_params,
        "STEREO_B": stereo_b_params
    }

    output_path = os.path.join(root, "output", "2_stereo_b_params.json")
    save_json(stereo_params, output_path)
    print("Stereo calibration parameters saved")
