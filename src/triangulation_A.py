import cv2
import numpy as np
import pandas as pd
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from matplotlib import pyplot as plt


def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")


def plot_projections_valid(
        cam1_reprojected_point,
        cam2_reprojected_point,
        row_data,
        matched_stereo_frames_folder,
        reprojected_frames_folder,
        scale_factor):
    frame_no = int(row_data['frame_no'])
    frame_folder = f"{matched_stereo_frames_folder}/frame_{frame_no}"
    cam1_frame = f"{frame_folder}/1_{frame_no}.jpg"
    cam2_frame = f"{frame_folder}/2_{frame_no}.jpg"

    cam1_frame = cv2.imread(cam1_frame)
    cam2_frame = cv2.imread(cam2_frame)

    def plot_reprojected_point(frame, point, color=(0, 255, 0), radius=15):
        frame_copy = frame.copy()
        # Convert point coordinates to integers and ensure they are within frame boundaries
        x = int(round(point[0] / scale_factor))
        y = int(round(point[1] / scale_factor))

        # Check if coordinates are within frame boundaries
        if frame_copy is not None and 0 <= x < frame_copy.shape[1] and 0 <= y < frame_copy.shape[0]:
            cv2.circle(frame_copy, (x, y), radius, color, -1)
        else:
            print(f"Warning: Point ({x}, {y}) is outside frame boundaries or frame is None")

        return frame_copy

    # Plot on both frames
    cam1_with_point = plot_reprojected_point(cam1_frame, cam1_reprojected_point)
    cam2_with_point = plot_reprojected_point(cam2_frame, cam2_reprojected_point)

    save_folder = f"{reprojected_frames_folder}/valid/frame_{frame_no}"
    os.makedirs(save_folder, exist_ok=True)
    cv2.imwrite(f"{save_folder}/cam1.jpg", cam1_with_point)
    cv2.imwrite(f"{save_folder}/cam2.jpg", cam2_with_point)


def plot_projection_anomaly(
        cam1_reprojected_point,
        cam2_reprojected_point,
        row_data,
        matched_stereo_frames_folder,
        reprojected_frames_folder,
        scale_factor):
    frame_no = int(row_data['frame_no'])
    frame_folder = f"{matched_stereo_frames_folder}/frame_{frame_no}"
    cam1_frame = f"{frame_folder}/1_{frame_no}.jpg"
    cam2_frame = f"{frame_folder}/2_{frame_no}.jpg"

    cam1_frame = cv2.imread(cam1_frame)
    cam2_frame = cv2.imread(cam2_frame)

    def plot_reprojected_point(frame, point, color=(0, 255, 0), radius=15):
        frame_copy = frame.copy()
        # Convert point coordinates to integers and ensure they are within frame boundaries
        x = int(round(point[0] / scale_factor))
        y = int(round(point[1] / scale_factor))

        # Check if coordinates are within frame boundaries
        if frame_copy is not None and 0 <= x < frame_copy.shape[1] and 0 <= y < frame_copy.shape[0]:
            cv2.circle(frame_copy, (x, y), radius, color, -1)
        else:
            print(f"Warning: Point ({x}, {y}) is outside frame boundaries or frame is None")

        return frame_copy

    # Plot on both frames
    cam1_with_point = plot_reprojected_point(cam1_frame, cam1_reprojected_point)
    cam2_with_point = plot_reprojected_point(cam2_frame, cam2_reprojected_point)

    save_folder = f"{reprojected_frames_folder}/anomalies/frame_{frame_no}"
    os.makedirs(save_folder, exist_ok=True)
    cv2.imwrite(f"{save_folder}/cam1.jpg", cam1_with_point)
    cv2.imwrite(f"{save_folder}/cam2.jpg", cam2_with_point)


def get_projection_matrices(
        K_cam1_undistort,
        K_cam2_undistort,
        stereo_rotation_matrix,
        stereo_translation_vector):
    """
    Computes the projection matrices for stereo triangulation.
    Camera 1 is set as the world coordinate system origin. Its projection matrix
    is computed using identity rotation and zero translation, while camera 2's
    projection matrix incorporates the stereo transformation parameters.
    """
    P1 = np.dot(K_cam1_undistort, np.hstack((np.eye(3), np.zeros((3, 1)))))
    R2 = np.array(stereo_rotation_matrix)
    T2 = np.array(stereo_translation_vector)
    P2 = np.dot(K_cam2_undistort, np.hstack((R2, T2)))
    return P1, P2


def get_average_error_cm(detections_csv):
    valid_detections = detections_csv[detections_csv['anomaly_detected'] == False]
    if len(valid_detections) == 0:
        return 0, 0
    a1_total_cm_error = valid_detections['A1_error_cm'].sum()
    average_cm_error_a1 = a1_total_cm_error / len(valid_detections)
    a2_total_cm_error = valid_detections['A2_error_cm'].sum()
    average_cm_error_a2 = a2_total_cm_error / len(valid_detections)
    return average_cm_error_a1, average_cm_error_a2


def get_average_cm_threshold(detections_csv, pixel_threshold, Fx_cam1, Fx_cam2):
    if len(detections_csv) == 0:
        return 0, 0

    # Get all valid 3D points and extract Z coordinates
    depths = []
    for points_3d_str in detections_csv['position_3d_pixels']:
        points_3d = np.array(eval(points_3d_str))
        depths.append(points_3d[2])

    avg_depth = np.mean(depths)

    threshold_cm_cam1 = (pixel_threshold * avg_depth) / (Fx_cam1 * 10)
    threshold_cm_cam2 = (pixel_threshold * avg_depth) / (Fx_cam2 * 10)

    return threshold_cm_cam1, threshold_cm_cam2


def preprocess_detections(detections_csv, threshold=0.3):
    # count how many detections are in the csv file (how many rows)
    detections = detections_csv.copy()
    previous_detection_count = len(detections)
    # Convert frame_no to integer
    detections['frame_no'] = detections['frame_no'].astype(int)
    # order detections by frame number
    detections = detections.sort_values(by='frame_no')
    # if A1_confidence or A2_confidence is less than 0.4, remove the detection
    detections = detections[detections['A1_confidence'] > threshold]
    detections = detections[detections['A2_confidence'] > threshold]
    filtered_detection_count = len(detections)
    return detections, filtered_detection_count, previous_detection_count


def compute_reprojection_error(points_3d, P1, P2, undistorted_point_camera_1, undistorted_point_camera_2):
    # Reproject 3D points (in mm) back to 2D
    reprojected_point_1 = np.dot(P1, np.vstack((points_3d, np.ones((1, points_3d.shape[1])))))
    reprojected_point_2 = np.dot(P2, np.vstack((points_3d, np.ones((1, points_3d.shape[1])))))

    reprojected_point_1 /= reprojected_point_1[2]
    reprojected_point_2 /= reprojected_point_2[2]

    # ravel() method is used to convert the 2D array into a 1D array
    cam1_reprojected_point = reprojected_point_1[:2].ravel()
    cam2_reprojected_point = reprojected_point_2[:2].ravel()

    # euclidean distance between the original and the reprojected point
    error1 = np.linalg.norm(cam1_reprojected_point - undistorted_point_camera_1.ravel())
    error2 = np.linalg.norm(cam2_reprojected_point - undistorted_point_camera_2.ravel())
    return error1, error2, cam1_reprojected_point, cam2_reprojected_point


def get_error_cm(error1, error2, cam1_fx, cam2_fx, points_3d):
    Z = points_3d[2][0]  # depth in mm
    error1_cm = (error1 * Z) / (cam1_fx * 10)
    error2_cm = (error2 * Z) / (cam2_fx * 10)
    return error1_cm, error2_cm


def triangulate(
    triangulated_detections_path,
    scale_factor,
    pixel_threshold,
    confidence_threshold,
    detections_csv,
    matched_stereo_frames_folder,
    reprojected_frames_folder,
    Fx_cam1,
    K_cam1_undistort,
    D_cam1,
    Fx_cam2,
    K_cam2_undistort,
    D_cam2,
    P1,
    P2
):
    detections_csv, after_count, prev_count = preprocess_detections(
        detections_csv, confidence_threshold)

    valid_detections = 0
    for index, row in detections_csv.iterrows():
        print(f"Processing frame {row['frame_no']}")
        # 2D points from YOLO, measure = pixels
        cam1_2d = (row['A1_x'] * scale_factor, row['A1_y'] * scale_factor)
        cam2_2d = (row['A2_x'] * scale_factor, row['A2_y'] * scale_factor)

        # undirtort the points
        undistorted_point_camera_1 = cv2.undistortPoints(
            cam1_2d, K_cam1_undistort, D_cam1, P=K_cam1_undistort)
        undistorted_point_camera_2 = cv2.undistortPoints(
            cam2_2d, K_cam2_undistort, D_cam2, P=K_cam2_undistort)

        # triangulate
        points_3d_homogeneous = cv2.triangulatePoints(
            P1, P2, undistorted_point_camera_1, undistorted_point_camera_2)

        # convert to 3D points
        points_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]

        # get back the 2D points from the 3D points to validate the triangulation
        # the reprojected points are 2D points after triangulation
        e1_px, e2_px, cam1_reprojected_point, cam2_reprojected_point = compute_reprojection_error(
            points_3d, P1, P2, undistorted_point_camera_1, undistorted_point_camera_2)

        # get the difference of original 2D with respect to the reprojected 2D in centimeters
        e1_cm, e2_cm = get_error_cm(e1_px, e2_px, Fx_cam1, Fx_cam2, points_3d)

        # using pixel threshold to check what is the error in cm
        if e1_px > pixel_threshold or e2_px > pixel_threshold:
            anomaly_detected = True
            print(
                f"----------------------------------Anomaly detected {row['frame_no']}----------------------------------")
            print(f"ORIGINAL A1: {tuple(map(lambda x: round(float(x), 2), cam1_2d))}")
            print(
                f"REPROJECTION A1: {tuple(map(lambda x: round(float(x), 2), cam1_reprojected_point))}")
            print(f"ORIGINAL A2: {tuple(map(lambda x: round(float(x), 2), cam2_2d))}")
            print(
                f"REPROJECTION A2: {tuple(map(lambda x: round(float(x), 2), cam2_reprojected_point))}")
            print(f"ERROR A1 (cm): {round(float(e1_cm), 2)}")
            print(f"ERROR A2 (cm): {round(float(e2_cm), 2)}")
            try:
                # save anomaly reprojection plots in a separate folder 
                plot_projection_anomaly(cam1_reprojected_point, cam2_reprojected_point,
                                        row, matched_stereo_frames_folder, reprojected_frames_folder, scale_factor)
            except Exception as e:
                print(f"Error saving anomaly plot for frame {row['frame_no']}: {e}")
        else:
            anomaly_detected = False
            valid_detections += 1
            try:
                # plot the 2D points extracted from the 3D points to visualise how close they are with respect to the
                # original ball in the frame image
                plot_projections_valid(cam1_reprojected_point, cam2_reprojected_point,
                                    row, matched_stereo_frames_folder, reprojected_frames_folder, scale_factor)
            except Exception as e:
                print(f"Error saving valid plot for frame {row['frame_no']}: {e}")

        # update CSV file with the new data
        points_3d_str = str(points_3d.ravel().tolist())
        cam1_reprojected_str = str(cam1_reprojected_point.tolist())
        cam2_reprojected_str = str(cam2_reprojected_point.tolist())
        detections_csv.at[index, 'position_3d_pixels'] = points_3d_str
        detections_csv.at[index, 'A1_reprojected_point'] = cam1_reprojected_str
        detections_csv.at[index, 'A1_error_cm'] = e1_cm
        detections_csv.at[index, 'A2_reprojected_point'] = cam2_reprojected_str
        detections_csv.at[index, 'A2_error_cm'] = e2_cm
        detections_csv.at[index, 'anomaly_detected'] = anomaly_detected

    # save the updated CSV file
    detections_csv.to_csv(triangulated_detections_path, index=False)
    print(f"Saved to {triangulated_detections_path}")

    # average error in centimeters of the reprojected points with respect to the original points
    a1_average_cm_error, a2_average_cm_error = get_average_error_cm(detections_csv)

    # the threshold we use is in pixels, we need to convert it to centimeters
    # just to know better what is the threshold we use
    # (convert the whole pipeline in centimeters it's more annoying to debug)
    A1_threshold_cm, A2_threshold_cm = get_average_cm_threshold(
        detections_csv, pixel_threshold, Fx_cam1, Fx_cam2)

    return {
        "confidence_threshold": confidence_threshold,
        "pixel_threshold": pixel_threshold,
        "total_raw_set": prev_count,
        "total_filtered_set": len(detections_csv),
        "valid_detections": valid_detections,
        "percentage_valid_detections": round((valid_detections / len(detections_csv)) * 100, 2),
        "A1_average_cm_error": round(float(a1_average_cm_error), 2),
        "A2_average_cm_error": round(float(a2_average_cm_error), 2),
        "A1_threshold_cm": round(float(A1_threshold_cm), 2),
        "A2_threshold_cm": round(float(A2_threshold_cm), 2)
    }


if __name__ == "__main__":
    root = find_project_root()
    detections_path = f"{root}/data/global_A.csv"
    detections_csv = pd.read_csv(detections_path, dtype={'frame_no': int})
    matched_stereo_frames_folder = f"{root}/global_matched_frames"
    triangulated_detections_path = f"{root}/data/global_A_triangulated.csv"
    reprojected_frames_folder = f"{root}/reprojected_global_A"
    
    if not os.path.exists(reprojected_frames_folder):
        os.makedirs(reprojected_frames_folder)
    else:
        shutil.rmtree(reprojected_frames_folder)
        os.makedirs(reprojected_frames_folder)

    # NOTE: get intrinsics and extrinsics parameters
    # get intrinsics parameters from json file
    intrinsics_path = f"{root}/output/V2_intrinsic_params.json"
    with open(intrinsics_path, 'r') as f:
        intrinsic_params = json.load(f)

    # get extrinsics parameters from json file
    extrinsics_path = f"{root}/output/V2_stereo_params.json"
    with open(extrinsics_path, 'r') as f:
        extrinsics = json.load(f)

    Fx_cam1 = intrinsic_params['CAM_1']['fx']
    K_cam1_undistort = np.array(intrinsic_params['CAM_1']['K_undistort'])
    D_cam1 = np.array(intrinsic_params['CAM_1']['D'])

    Fx_cam2 = intrinsic_params['CAM_2']['fx']
    K_cam2_undistort = np.array(intrinsic_params['CAM_2']['K_undistort'])
    D_cam2 = np.array(intrinsic_params['CAM_2']['D'])

    stereo_rotation_matrix = np.array(extrinsics['STEREO_A']['rotation_matrix'])
    stereo_translation_vector = np.array(extrinsics['STEREO_A']['translation_vector'])

    P1, P2 = get_projection_matrices(
        K_cam1_undistort,
        K_cam2_undistort,
        stereo_rotation_matrix,
        stereo_translation_vector)

    # yolo points were taken in 1080 frames, and triangulation is done in 4K
    # so we need to scale up the points
    scale_factor = 2
    # threshold to filter YOLO bad detections
    confidence_threshold = 0.2
    # threshold to detect anomalies
    # if the error between the original and the reprojected point is greater than this threshold,
    # we consider the detection as an anomaly
    pixel_threshold = 80  # 80px = 30cm approx
    results_data = triangulate(
        triangulated_detections_path,
        scale_factor,
        pixel_threshold,
        confidence_threshold,
        detections_csv,
        matched_stereo_frames_folder,
        reprojected_frames_folder,
        Fx_cam1,
        K_cam1_undistort,
        D_cam1,
        Fx_cam2,
        K_cam2_undistort,
        D_cam2,
        P1,
        P2
    )
    pprint(results_data)
    # save results as json file
    with open(f"{root}/output/triangulation_result_summary/global_triangulation_A.json", 'w') as f:
        json.dump(results_data, f)
    print("triangulation done")
