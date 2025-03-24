import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import sys

# imread_img = cv.imread(image_path)


def print_image(imread_img):
    img = cv2.cvtColor(imread_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def get_frame_size(image_path_pattern):
    image_list = glob.glob(image_path_pattern)
    if not image_list:
        raise FileNotFoundError("No images found")
    img = cv2.imread(image_list[0])
    height, width = img.shape[:2]
    return (width, height)


def compute_full_stereo_reprojection_error_for_pair(
    groups_folder, 
    chessboard_size, 
    square_size_mm, 
    intrinsics, 
    full_stereo_params, 
    left_cam_key, 
    right_cam_key, letter="a"):
    """
    Computes the reprojection error for a specified pair of cameras.
    For each group:
      1. Detect the chessboard in the left and right images.
      2. Estimate the pose (rvec, tvec) for the left camera using its intrinsics.
      3. Apply the full stereo calibration (relative transformation) to predict the right camera's pose.
      4. Project the 3D chessboard points into the right image.
      5. Compare the projected points with the detected corners.
    The function returns the average reprojection error over all valid groups.
    """
    # Prepare 3D object points for the chessboard
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    group_folders = sorted(glob.glob(os.path.join(groups_folder, "*")))
    errors = []

    for group in group_folders:
        # Construct image paths for the given left and right cameras
        # Expecting file names to be lower-case versions of the camera keys (e.g., "left_a.png", "right_a.png")
        img_left = os.path.join(group, f"left_{letter}.png")
        img_right = os.path.join(group, f"right_{letter}.png")

        if not (os.path.exists(img_left) and os.path.exists(img_right)):
            print("no images found")
            continue

        # Read images
        im_left = cv2.imread(img_left)
        im_right = cv2.imread(img_right)

        # Detect chessboard corners in both images
        ret_left, corners_left = detect_chessboard(im_left, chessboard_size)
        ret_right, corners_right = detect_chessboard(im_right, chessboard_size)
        if not (ret_left and ret_right):
            continue

        # Solve pose for the left camera using its intrinsics
        K_left = np.array(intrinsics[left_cam_key]["K"])
        D_left = np.array(intrinsics[left_cam_key]["D"])
        ret_pose, rvec_left, tvec_left = cv2.solvePnP(objp, corners_left, K_left, D_left)
        if not ret_pose:
            continue

        # Get full stereo calibration parameters (relative transformation from left to right)
        full_rvec = np.array(full_stereo_params["rotation_vector"]).reshape(3, 1)
        full_tvec = np.array(full_stereo_params["translation_vector"]).reshape(3, 1)
        R_full, _ = cv2.Rodrigues(full_rvec)

        # Compute predicted right camera pose from the left camera pose
        R_left, _ = cv2.Rodrigues(rvec_left)
        R_pred = R_full @ R_left
        t_pred = full_tvec + R_full @ tvec_left
        rvec_pred, _ = cv2.Rodrigues(R_pred)

        # Project the 3D chessboard points into the right image using its intrinsics
        K_right = np.array(intrinsics[right_cam_key]["K"])
        D_right = np.array(intrinsics[right_cam_key]["D"])
        proj_points, _ = cv2.projectPoints(objp, rvec_pred, t_pred, K_right, D_right)

        # Compute the reprojection error (average L2 distance)
        error = cv2.norm(corners_right, proj_points, cv2.NORM_L2) / len(objp)
        errors.append(error)

    if errors:
        avg_error = round(np.mean(errors), 3)
        print(
            f"Full stereo reprojection error for {left_cam_key} -> {right_cam_key}: {avg_error}")
        return avg_error
    else:
        print(
            f"No valid groups found for computing full stereo reprojection error for {left_cam_key} -> {right_cam_key}.")
        return None
