import cv2
import numpy as np
import os
import glob
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from utils.utils import print_image

# TODO: for LEFT_CAM_B ↔ RIGHT_CAM_B compute a separate full_stereo transform for those

"""
To solve aligment problem to make both stereos be in the same coordinate system
- find the new rotation matrix and translation vector
- this is just calibrating cam A from stereo 1 with cam A from stereo 2

HOW TO IMPROVE FULL CALIBRATION:
1. Global bundle adjustment
  - optimization process using a cost function to minimise reprojection error

2. Using known scene constraints
  - use tennis court for the full calibration
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


def detect_chessboard(frame, pattern_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # use adaptive threshold and normalization flags for better detection
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        # stricter termination criteria for improved subpixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return ret, corners


def apply_global_transformation_to_extrinsics(extrinsics, full_stereo_params):
    # full_stereo_params computed from left A to right A.
    # We invert it to map right extrinsics into left system.
    R_full, _ = cv2.Rodrigues(np.array(full_stereo_params["rotation_vector"]))
    t_full = np.array(full_stereo_params["translation_vector"]).reshape(3, 1)
    R_inv = R_full.T
    t_inv = -R_inv @ t_full

    updated_extrinsics = {}
    for cam, params in extrinsics.items():
        R_cam, _ = cv2.Rodrigues(np.array(params["rvec"]))
        t_cam = np.array(params["tvec"]).reshape(3, 1)
        # New extrinsics: T_new = T_inv * T_cam
        R_new = R_inv @ R_cam
        t_new = R_inv @ (t_cam - t_full)
        rvec_new, _ = cv2.Rodrigues(R_new)
        updated_extrinsics[cam] = {"rvec": rvec_new.flatten().tolist(),
                                   "tvec": t_new.flatten().tolist()}
    return updated_extrinsics


def calibrate_full_stereo_from_groups(groups_folder, chessboard_size, square_size_mm, intrinsics, stereo_extrinsics):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    group_folders = sorted(glob.glob(os.path.join(groups_folder, "*")))
    print(group_folders)

    left_a_extrinsics = []
    right_a_extrinsics = []

    for group in tqdm(group_folders, desc="Processing groups"):
        print("GROUP -> ", group)
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

        # make sure the chessboard is visible in all cameras
        ret_la, corners_la = detect_chessboard(im_left_a, chessboard_size)
        ret_lb, corners_lb = detect_chessboard(im_left_b, chessboard_size)
        ret_ra, corners_ra = detect_chessboard(im_right_a, chessboard_size)
        ret_rb, corners_rb = detect_chessboard(im_right_b, chessboard_size)

        if not (ret_la and ret_lb and ret_ra and ret_rb):
            print("chessboard not found in one of the frames within a group")
            continue


        K_la = np.array(intrinsics["LEFT_CAM_A"]["K"])
        D_la = np.array(intrinsics["LEFT_CAM_A"]["D"])

        K_ra = np.array(intrinsics["RIGHT_CAM_A"]["K"])
        D_ra = np.array(intrinsics["RIGHT_CAM_A"]["D"])

        # position of the chessboard for each camera A
        # gives the pose of the chessboard in each camera coordinate system
        # TODO: check reprojection error
        ret_la, rvec_la, tvec_la = cv2.solvePnP(objp, corners_la, K_la, D_la)
        ret_ra, rvec_ra, tvec_ra = cv2.solvePnP(objp, corners_ra, K_ra, D_ra)

        if not (ret_la and ret_ra):
            print("solvePnP not working")
            continue

        left_a_extrinsics.append((rvec_la, tvec_la))
        right_a_extrinsics.append((rvec_ra, tvec_ra))

    if len(left_a_extrinsics) == 0:
        print(left_a_extrinsics)
        raise Exception("No valid groups found for full stereo calibration.")

    # Compute relative transformation from left_a to right_a for each group:
    rel_rvecs = []
    rel_tvecs = []
    for (rvec_la, tvec_la), (rvec_ra, tvec_ra) in zip(left_a_extrinsics, right_a_extrinsics):
        # convert rotation vectors from cameras A to rotation matrices (R)
        R_la, _ = cv2.Rodrigues(rvec_la)
        R_ra, _ = cv2.Rodrigues(rvec_ra)
        # flip left camera A position to get the camera position from the chessboard point of view
        R_la_inv = R_la.T
        tvec_la_inv = -R_la_inv @ tvec_la
        # combine both views to know how to move from left A to right A
        # Relative transformation: T = T_right_a * inv(T_left_a)
        R_rel = R_ra @ R_la_inv
        tvec_rel = tvec_ra + R_ra @ tvec_la_inv
        # convert R back to vector
        rvec_rel, _ = cv2.Rodrigues(R_rel)

        rel_rvecs.append(rvec_rel)
        rel_tvecs.append(tvec_rel)

    avg_rvec_groups = np.mean(np.array(rel_rvecs), axis=0)
    avg_tvec_groups = np.mean(np.array(rel_tvecs), axis=0)

    full_stereo_params = {
        "rotation_vector": avg_rvec_groups.flatten().tolist(),
        "translation_vector": avg_tvec_groups.flatten().tolist(),
        "num_groups_used": len(rel_rvecs)
    }
    print("Full stereo calibration completed using", len(rel_rvecs), "groups.")
    return full_stereo_params

if __name__ == "__main__":
    root = find_project_root()
    groups_folder = os.path.join(root, "images", "full-stereo", "GROUPS")

    with open(os.path.join(root, "output", "intrinsic_params.json"), "r") as f:
        intrinsics = json.load(f)
    with open(os.path.join(root, "output", "keep_best_stereo_params.json"), "r") as f:
        stereo_extrinsics = json.load(f)

    chessboard_size = (9, 6)
    square_size_mm = 60

    # Compute global transformation using left_a and right_a images.
    full_stereo_params = calibrate_full_stereo_from_groups(
        groups_folder, chessboard_size, square_size_mm, intrinsics, stereo_extrinsics)
    print("Global transformation:", full_stereo_params)

    # Update RIGHT cameras into the LEFT coordinate system.
    updated_right = apply_global_transformation_to_extrinsics({
        "RIGHT_CAM_A": stereo_extrinsics["RIGHT_CAM_A"],
        "RIGHT_CAM_B": stereo_extrinsics["RIGHT_CAM_B"]
    }, full_stereo_params)

    # Global extrinsics for all cameras.
    global_extrinsics = {
        "LEFT_CAM_A": stereo_extrinsics["LEFT_CAM_A"],
        "LEFT_CAM_B": stereo_extrinsics["LEFT_CAM_B"],
        "RIGHT_CAM_A": updated_right["RIGHT_CAM_A"],
        "RIGHT_CAM_B": updated_right["RIGHT_CAM_B"]
    }

    # Save global extrinsics.
    output_path = os.path.join(root, "output", "global_extrinsics.json")
    save_json(global_extrinsics, output_path)
    print("Global extrinsics saved to", output_path)
