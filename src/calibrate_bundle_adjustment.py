import cv2
import numpy as np
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import least_squares

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

def rodrigues_to_vec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()

def project_points(objp, rvec, tvec, K):
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Transform object points to camera coordinates
    X_cam = (R @ objp.T).T + tvec.reshape(1, 3)
    # Perspective division
    x = X_cam[:, 0] / X_cam[:, 2]
    y = X_cam[:, 1] / X_cam[:, 2]
    # Apply intrinsics (no distortion considered here)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x + cx
    v = fy * y + cy
    return np.column_stack([u, v])

def bundle_adjustment_residuals(params, n_cams, n_groups, camera_indices, group_indices, 
                                observations, objp, intrinsics_list):
    """
    params: parameter vector containing camera parameters and group (chessboard) poses.
    n_cams: number of cameras (should be 4).
    n_groups: number of groups (each group is one chessboard pose).
    camera_indices: list mapping each observation to a camera index.
    group_indices: list mapping each observation to a group index.
    observations: list of observed 2D points for each observation (shape: (n_points, 2)).
    objp: (N,3) array of chessboard 3D points in the chessboard coordinate system.
    intrinsics_list: list of K matrices corresponding to each camera (order matches camera_indices).
    """
    residuals = []
    # Extract camera parameters: each camera has 6 parameters [rvec (3), tvec (3)]
    camera_params = params[:n_cams * 6].reshape((n_cams, 6))
    # Extract group parameters: each group has 6 parameters [rvec (3), tvec (3)]
    group_params = params[n_cams * 6:].reshape((n_groups, 6))
    
    for i, obs in enumerate(observations):
        cam_idx = camera_indices[i]
        grp_idx = group_indices[i]
        # Get camera parameters
        rvec_cam = camera_params[cam_idx, :3]
        tvec_cam = camera_params[cam_idx, 3:]
        # Get group (chessboard) pose parameters
        rvec_grp = group_params[grp_idx, :3]
        tvec_grp = group_params[grp_idx, 3:]
        # First, transform the chessboard object points to world coordinates
        R_grp, _ = cv2.Rodrigues(rvec_grp)
        X_world = (R_grp @ objp.T).T + tvec_grp.reshape(1, 3)
        # Now, project these world points into the image using the camera pose
        R_cam, _ = cv2.Rodrigues(rvec_cam)
        X_cam = (R_cam @ X_world.T).T + tvec_cam.reshape(1, 3)
        # Perspective division
        x = X_cam[:, 0] / X_cam[:, 2]
        y = X_cam[:, 1] / X_cam[:, 2]
        # Get camera intrinsics
        K = intrinsics_list[cam_idx]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u_proj = fx * x + cx
        v_proj = fy * y + cy
        proj_points = np.column_stack([u_proj, v_proj])
        # Residual: difference between projected points and observed points
        res = (proj_points - obs).flatten()
        residuals.append(res)
    return np.concatenate(residuals)

def perform_bundle_adjustment(groups_folder, intrinsics, init_camera_extrinsics, chessboard_size, square_size_mm):
    """
    This function performs global bundle adjustment for your multi-stereo system.
    It uses the fixed intrinsics and initial camera extrinsics (for all 4 cameras) as a starting point.
    It also computes an initial guess for each group’s (chessboard) pose from the left_a image.
    
    Returns:
        refined_camera_extrinsics: dict with keys for each camera and their refined extrinsics.
    """
    # Prepare the 3D object points for the chessboard pattern
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm  # scale by square size

    # Define the camera order; these will be used to index parameters and intrinsics.
    camera_names = ["LEFT_CAM_A", "LEFT_CAM_B", "RIGHT_CAM_A", "RIGHT_CAM_B"]
    n_cams = len(camera_names)
    # Build list of camera intrinsics matrices in the same order.
    intrinsics_list = []
    for cam in camera_names:
        K = np.array(intrinsics[cam]["K"])
        intrinsics_list.append(K)

    # Prepare lists for observations:
    # For each valid group and each camera, store:
    # - The observed 2D points (detected chessboard corners)
    # - Which camera (index) this observation comes from.
    # - Which group index (each group corresponds to one chessboard pose).
    observations = []     # list of arrays (n_points, 2)
    camera_indices = []   # list of int (0 to n_cams-1)
    group_indices = []    # list of int (0 to n_groups-1)
    group_initial_poses = []  # initial guess for each group's pose: [rvec (3), tvec (3)]
    group_folders = sorted(glob.glob(os.path.join(groups_folder, "*")))
    
    valid_group_count = 0
    for group in tqdm(group_folders, desc="Collecting observations"):
        # File names for the four cameras in this group
        file_left_a = os.path.join(group, "left_a.png")
        file_left_b = os.path.join(group, "left_b.png")
        file_right_a = os.path.join(group, "right_a.png")
        file_right_b = os.path.join(group, "right_b.png")
        if not (os.path.exists(file_left_a) and os.path.exists(file_left_b) and
                os.path.exists(file_right_a) and os.path.exists(file_right_b)):
            continue

        # Read images
        im_left_a = cv2.imread(file_left_a)
        im_left_b = cv2.imread(file_left_b)
        im_right_a = cv2.imread(file_right_a)
        im_right_b = cv2.imread(file_right_b)

        # Detect chessboard in each image
        ret_la, corners_la = detect_chessboard(im_left_a, chessboard_size)
        ret_lb, corners_lb = detect_chessboard(im_left_b, chessboard_size)
        ret_ra, corners_ra = detect_chessboard(im_right_a, chessboard_size)
        ret_rb, corners_rb = detect_chessboard(im_right_b, chessboard_size)

        if not (ret_la and ret_lb and ret_ra and ret_rb):
            print(f"Chessboard not detected in all images in group: {group}")
            continue

        # For each camera, add the detected corners as an observation.
        # Order must match camera_names.
        observations.append(corners_la.reshape(-1, 2))
        camera_indices.append(camera_names.index("LEFT_CAM_A"))
        observations.append(corners_lb.reshape(-1, 2))
        camera_indices.append(camera_names.index("LEFT_CAM_B"))
        observations.append(corners_ra.reshape(-1, 2))
        camera_indices.append(camera_names.index("RIGHT_CAM_A"))
        observations.append(corners_rb.reshape(-1, 2))
        camera_indices.append(camera_names.index("RIGHT_CAM_B"))
        
        # Compute an initial guess for the chessboard (group) pose.
        # Here, we use the left_a image’s solvePnP result.
        K_la = np.array(intrinsics["LEFT_CAM_A"]["K"])
        D_la = np.array(intrinsics["LEFT_CAM_A"]["D"])
        ret_pnp, rvec_grp, tvec_grp = cv2.solvePnP(objp, corners_la, K_la, D_la)
        if not ret_pnp:
            print(f"solvePnP failed for group: {group}")
            continue
        group_initial_poses.append(np.hstack([rvec_grp.flatten(), tvec_grp.flatten()]))
        
        # For each observation in this group, record the same group index.
        for _ in range(4):
            group_indices.append(valid_group_count)
        valid_group_count += 1

    n_groups = valid_group_count
    if n_groups == 0:
        raise Exception("No valid groups with chessboard detections found for bundle adjustment.")

    # Build initial parameter vector.
    # Camera parameters: use init_camera_extrinsics for each camera.
    camera_params = []
    for cam in camera_names:
        # Assume init_camera_extrinsics[cam] has keys "rvec" and "tvec" as lists.
        rvec = np.array(init_camera_extrinsics[cam]["rvec"]).flatten()
        tvec = np.array(init_camera_extrinsics[cam]["tvec"]).flatten()
        camera_params.append(np.hstack([rvec, tvec]))
    camera_params = np.array(camera_params)  # shape (n_cams, 6)

    # Group (chessboard pose) parameters.
    group_params = np.array(group_initial_poses)  # shape (n_groups, 6)

    # Concatenate into one parameter vector.
    x0 = np.hstack([camera_params.flatten(), group_params.flatten()])

    print("Starting bundle adjustment optimization...")
    res = least_squares(bundle_adjustment_residuals, x0, verbose=2, x_scale='jac',
                        ftol=1e-4, method='trf',
                        args=(n_cams, n_groups, camera_indices, group_indices, observations, objp, intrinsics_list))
    print("Optimization complete.")

    # Extract refined camera parameters.
    refined_camera_params = res.x[:n_cams * 6].reshape((n_cams, 6))
    refined_extrinsics = {}
    for i, cam in enumerate(camera_names):
        refined_extrinsics[cam] = {
            "rvec": refined_camera_params[i, :3].tolist(),
            "tvec": refined_camera_params[i, 3:].tolist()
        }
    return refined_extrinsics

if __name__ == "__main__":
    # Find project root and define paths.
    root = find_project_root()
    groups_folder = os.path.join(root, "images", "full-stereo", "GROUPS")

    # Load intrinsics (per camera). Format as described.
    intrinsic_path = os.path.join(root, "output", "intrinsic_params.json")
    with open(intrinsic_path, "r") as f:
        intrinsics = json.load(f)

    # Load initial extrinsics for each camera.
    # For this example, we assume you have a JSON file with keys for each camera:
    # "LEFT_CAM_A", "LEFT_CAM_B", "RIGHT_CAM_A", "RIGHT_CAM_B"
    init_extrinsics_path = os.path.join(root, "output", "global_extrinsics_initial.json")
    with open(init_extrinsics_path, "r") as f:
        init_camera_extrinsics = json.load(f)

    chessboard_size = (9, 6)
    square_size_mm = 60

    # Perform global bundle adjustment.
    refined_extrinsics = perform_bundle_adjustment(groups_folder, intrinsics, init_camera_extrinsics, chessboard_size, square_size_mm)

    # Save the refined camera extrinsics for triangulation.
    output_path = os.path.join(root, "output", "bundle_adjusted_extrinsics.json")
    save_json(refined_extrinsics, output_path)
    print("Bundle adjusted extrinsics saved to", output_path)


"""
Intrincis format:
{
  "LEFT_CAM_A": {
    "fx": 1664.262223296632,
    "fy": 1674.0224375413281,
    "cx": 1909.3164410278362,
    "cy": 1060.9036722529374,
    "D": [
      -0.060868117525704486, 0.11893053236468289, -0.001257902014961629,
      -0.0056969580572269985, -0.09266167681992382
    ],
    "K": [
      [1664.262223296632, 0.0, 1909.3164410278362],
      [0.0, 1674.0224375413281, 1060.9036722529374],
      [0.0, 0.0, 1.0]
    ],
    "reprojection_error": 0.712,
    "K_undistort": [
      [1366.0275575743908, 0.0, 1567.168225093463],
      [0.0, 1060.7128109731423, 843.3235010168614],
      [0.0, 0.0, 1.0]
    ],
    "roi_undistort": [0, 171, 3151, 1368],
    "count_bfore": 50,
    "count_after": 26
  },
  "LEFT_CAM_B": {
    "fx": 1621.1361689509747,
    "fy": 1625.1603810515107,
    "cx": 1965.154008519381,
    "cy": 1084.978764891054,
    "D": [
      0.021627463383018804, -0.12821638663437315, 0.0012833264234245073,
      0.0026517079736738847, 0.18254838980834961
    ],
    "K": [
      [1621.1361689509747, 0.0, 1965.154008519381],
      [0.0, 1625.1603810515107, 1084.978764891054],
      [0.0, 0.0, 1.0]
    ],
    "reprojection_error": 0.791,
    "K_undistort": [
      [1921.1070535728504, 0.0, 1904.1976159993976],
      [0.0, 1625.2998508696437, 1087.6311963706737],
      [0.0, 0.0, 1.0]
    ],
    "roi_undistort": [628, 493, 2654, 1184],
    "count_bfore": 35,
    "count_after": 25
  },
  "RIGHT_CAM_A": {
    "fx": 1587.2795966688107,
    "fy": 1603.2282576081725,
    "cx": 1940.9553202358818,
    "cy": 1008.9601301110795,
    "D": [
      0.12187697862757517, -0.5055972836008285, -0.01531407793829485,
      0.01229842564071875, 0.5589242417591878
    ],
    "K": [
      [1587.2795966688107, 0.0, 1940.9553202358818],
      [0.0, 1603.2282576081725, 1008.9601301110795],
      [0.0, 0.0, 1.0]
    ],
    "reprojection_error": 0.317,
    "K_undistort": [
      [2211.64187682825, 0.0, 1978.8473832395946],
      [0.0, 1598.3385276842532, 979.8011269264415],
      [0.0, 0.0, 1.0]
    ],
    "roi_undistort": [1195, 667, 1571, 653],
    "count_bfore": 50,
    "count_after": 28
  },
  "RIGHT_CAM_B": {
    "fx": 1661.3588559815864,
    "fy": 1651.770076375925,
    "cx": 1830.7438344899965,
    "cy": 1077.6890251436012,
    "D": [
      -0.08043249866394915, 0.0783679856228353, -0.001872061179847021,
      -0.012652129577415237, -0.0460508000660553
    ],
    "K": [
      [1661.3588559815864, 0.0, 1830.7438344899965],
      [0.0, 1651.770076375925, 1077.6890251436012],
      [0.0, 0.0, 1.0]
    ],
    "reprojection_error": 0.64,
    "K_undistort": [
      [1384.7581946187086, 0.0, 1754.789510196781],
      [0.0, 1116.6393932204492, 992.1700340602727],
      [0.0, 0.0, 1.0]
    ],
    "roi_undistort": [229, 264, 3200, 1460],
    "count_bfore": 50,
    "count_after": 25
  }
}

extrinsic json format:
{
  "stereo_left": {
    "rotation_matrix": [
      [0.44685186797011744, -0.01696136060367611, 0.8944471590530593],
      [-0.014830425002661239, 0.9995424120642284, 0.026363326403634636],
      [-0.894485028709736, -0.025045533160549863, 0.44639584975988145]
    ],
    "translation_vector": [
      [182.79791549055133],
      [3.656006096936827],
      [-221.99213146508572]
    ],
    "essential_matrix": [
      [-6.562480375455376, 221.79898392195508, 7.484476969222048],
      [64.31240006580427, 8.343559847055989, -280.1604621408188],
      [-4.344663930051788, 182.77628020751806, 1.549056845095616]
    ],
    "fundamental_matrix": [
      [-8.603389787980885e-9, 2.890823755227712e-7, -0.0002739320125124575],
      [8.410456486592994e-8, 1.084767629694079e-8, -0.0007818428741137033],
      [-8.357841298713144e-5, -0.0001936705598246269, 1.0]
    ],
    "reprojection_error": 0.5839885339141655,
    "num_valid_pairs": 25
  },
  "stereo_right": {
    "rotation_matrix": [
      [0.44244884250309907, 0.0490364566523282, -0.8954520912291467],
      [-0.04013237827210425, 0.9985862758568937, 0.03485458197259439],
      [0.8958953141868152, 0.020515252600051152, 0.44379129151985797]
    ],
    "translation_vector": [
      [-135.20219484598928],
      [-12.200506904189856],
      [-510.14989526891475]
    ],
    "essential_matrix": [
      [-31.403905538574037, 509.1783875643818, 12.366582626753473],
      [-104.58821783451398, -22.242236046197103, 516.8163472259009],
      [10.824085784487856, -134.4127866109908, -15.637385404546661]
    ],
    "fundamental_matrix": [
      [1.5489125686830455e-7, -2.486401607023656e-6, 0.002111227248557311],
      [5.188476583000669e-7, 1.0924300009641726e-7, -0.005186835566069659],
      [-0.0009314173914081702, 0.005524683413595285, 1.0]
    ],
    "reprojection_error": 0.9161362747557085,
    "num_valid_pairs": 11
  }
}


"""