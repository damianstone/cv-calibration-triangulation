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
HOW TO IMPROVE FULL CALIBRATION:
1. Global bundle adjustment
  - optimization process using a cost function to minimise reprojection error

2. Using known scene constraints
  - use tennis court for the full calibration
"""


"""
Okay lets proceed with creating the pipeline to calibrate the full stereo using the groups of images I have.
The input folder is root/cameras/full-stereo/GROUPS/ - inside groups there are subfolder which sinde them contains
photos of the 4 cameras named:
- left_a.png
- left_b.png
- right_a.png
- right_b.png

the intrinsics are in: root/output/intrinsic_params.json
the extrinsic for each stereo pair are in root/output/keep_best_stereo_params.json

I want to create a similar pipeline as the one above, but to calibrate both stereo pair to form a full stereo system with
the 4 cameras. 

Here is an example pipeline style:

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
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    return ret, corners

def calibrate_full_stereo_from_groups(groups_folder, chessboard_size, square_size_mm, intrinsics):
    # Prepare 3D object points for the chessboard
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    group_folders = sorted(glob.glob(os.path.join(groups_folder, "*")))
    left_a_extrinsics = []
    right_a_extrinsics = []
    
    for group in group_folders:
        # Construct image paths
        img_left_a = os.path.join(group, "left_a.png")
        img_left_b = os.path.join(group, "left_b.png")
        img_right_a = os.path.join(group, "right_a.png")
        img_right_b = os.path.join(group, "right_b.png")
        if not (os.path.exists(img_left_a) and os.path.exists(img_left_b) and
                os.path.exists(img_right_a) and os.path.exists(img_right_b)):
            continue

        # Read images
        im_left_a = cv2.imread(img_left_a)
        im_left_b = cv2.imread(img_left_b)
        im_right_a = cv2.imread(img_right_a)
        im_right_b = cv2.imread(img_right_b)

        # (Optional) Check frame sizes with check_frame_size if desired

        # Detect chessboard corners (using your tuned detect_chessboard)
        ret_la, corners_la = detect_chessboard(im_left_a, chessboard_size)
        ret_lb, corners_lb = detect_chessboard(im_left_b, chessboard_size)
        ret_ra, corners_ra = detect_chessboard(im_right_a, chessboard_size)
        ret_rb, corners_rb = detect_chessboard(im_right_b, chessboard_size)
        if not (ret_la and ret_lb and ret_ra and ret_rb):
            continue

        # (Optionally, you can run an extra cornerSubPix here using your preferred winSize/criteria)

        # Compute poses for left_a and right_a (you can also use left_b/right_b if desired)
        K_la = np.array(intrinsics["LEFT_CAM_A"]["K"])
        D_la = np.array(intrinsics["LEFT_CAM_A"]["D"])
        K_ra = np.array(intrinsics["RIGHT_CAM_A"]["K"])
        D_ra = np.array(intrinsics["RIGHT_CAM_A"]["D"])

        ret_la, rvec_la, tvec_la = cv2.solvePnP(objp, corners_la, K_la, D_la)
        ret_ra, rvec_ra, tvec_ra = cv2.solvePnP(objp, corners_ra, K_ra, D_ra)
        if not (ret_la and ret_ra):
            continue

        left_a_extrinsics.append((rvec_la, tvec_la))
        right_a_extrinsics.append((rvec_ra, tvec_ra))
    
    if len(left_a_extrinsics) == 0:
        raise Exception("No valid groups found for full stereo calibration.")

    # Compute relative transformation from left_a to right_a for each group.
    # That is, T = T_right_a * inv(T_left_a)
    rel_rvecs = []
    rel_tvecs = []
    for (rvec_la, tvec_la), (rvec_ra, tvec_ra) in zip(left_a_extrinsics, right_a_extrinsics):
        R_la, _ = cv2.Rodrigues(rvec_la)
        R_ra, _ = cv2.Rodrigues(rvec_ra)
        # Inverse of left_a pose: inv(T_left_a) = [R_la.T, -R_la.T @ tvec_la]
        R_la_inv = R_la.T
        tvec_la_inv = -R_la_inv @ tvec_la
        # Relative transformation: T = T_right_a * inv(T_left_a)
        R_rel = R_ra @ R_la_inv
        tvec_rel = tvec_ra + R_ra @ tvec_la_inv
        rvec_rel, _ = cv2.Rodrigues(R_rel)
        rel_rvecs.append(rvec_rel)
        rel_tvecs.append(tvec_rel)
    
    # Average the relative rotations and translations (simple averaging)
    avg_rvec = np.mean(np.array(rel_rvecs), axis=0)
    avg_tvec = np.mean(np.array(rel_tvecs), axis=0)
    
    # Optionally, compute the reprojection error over all groups using the averaged parameters.
    # For simplicity, here we just return the averaged extrinsics.
    full_stereo_params = {
        "rotation_vector": avg_rvec.flatten().tolist(),
        "translation_vector": avg_tvec.flatten().tolist(),
        "num_groups_used": len(rel_rvecs)
    }
    print("Full stereo calibration completed using", len(rel_rvecs), "groups.")
    return full_stereo_params


if __name__ == "__main__":
    root = find_project_root()
    groups_folder = os.path.join(root, "cameras", "full-stereo", "GROUPS")
    
    # Load intrinsics
    intrinsic_path = os.path.join(root, "output", "intrinsic_params.json")
    with open(intrinsic_path, "r") as f:
        intrinsics = json.load(f)
    
    # Load extrinsics for each stereo pair if needed
    extrinsic_path = os.path.join(root, "output", "keep_best_stereo_params.json")
    with open(extrinsic_path, "r") as f:
        stereo_extrinsics = json.load(f)
    
    chessboard_size = (9, 6)
    square_size_mm = 60

    full_stereo_params = calibrate_full_stereo_from_groups(groups_folder, chessboard_size, square_size_mm, intrinsics)
    output_path = os.path.join(root, "output", "full_stereo_params.json")
    save_json(full_stereo_params, output_path)
    print("Full stereo calibration parameters saved to", output_path)
