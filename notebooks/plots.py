import cv2
import numpy as np
import pandas as pd
import os
import glob
import json
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")


def plot_2d_reprojected_points(triangulation_results, stereo='A', camera_names=['Camera 1', 'Camera 2']):
    """
    To plot the reprojected 2D points, extracted after triangulation. These are the 3D triangulated points without the Z axis.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Extract points
    points_2d_1 = np.array([eval(p)
                           for p in triangulation_results[f'{stereo}1_reprojected_point']])
    points_2d_2 = np.array([eval(p)
                           for p in triangulation_results[f'{stereo}2_reprojected_point']])
    anomaly_mask = triangulation_results['anomaly_detected']

    # Plot for camera 1
    ax1.scatter(points_2d_1[~anomaly_mask, 0],
                points_2d_1[~anomaly_mask, 1], c='blue', alpha=0.6, label='Normal')
    ax1.scatter(points_2d_1[anomaly_mask, 0], points_2d_1[anomaly_mask,
                1], c='red', alpha=0.6, label='Anomaly')
    ax1.set_title(f'{camera_names[0]} Reprojected Points')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)

    # Plot for camera 2
    ax2.scatter(points_2d_2[~anomaly_mask, 0],
                points_2d_2[~anomaly_mask, 1], c='blue', alpha=0.6, label='Normal')
    ax2.scatter(points_2d_2[anomaly_mask, 0], points_2d_2[anomaly_mask,
                1], c='red', alpha=0.6, label='Anomaly')
    ax2.set_title(f'{camera_names[1]} Reprojected Points')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_valid_vs_original(triangulation_results, stereo='A', camera_names=['Camera 1', 'Camera 2']):
    """
    Plot the original 2D YOLO points vs triangulated 2D points
    The idea is to see how accurate the triangulation is.
    """
    if len(camera_names) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))

    # calibration params are in 4K
    # YOLO points are in 1080, so we need to scale up the YOLO points
    scale_factor = 2
    # Extract points
    original_2d_1 = np.array([[row[f'{stereo}1_x'] * scale_factor, row[f'{stereo}1_y']
                             * scale_factor] for _, row in triangulation_results.iterrows()])
    points_2d_1 = np.array([eval(p)
                           for p in triangulation_results[f'{stereo}1_reprojected_point']])
    original_2d_2 = np.array([[row[f'{stereo}2_x'] * scale_factor, row[f'{stereo}2_y']
                             * scale_factor] for _, row in triangulation_results.iterrows()])
    points_2d_2 = np.array([eval(p)
                           for p in triangulation_results[f'{stereo}2_reprojected_point']])
    anomaly_mask = triangulation_results['anomaly_detected']

    # Plot for camera 1
    if len(camera_names) > 1:
        ax1.scatter(original_2d_1[~anomaly_mask, 0],
                    original_2d_1[~anomaly_mask, 1], c='blue', alpha=0.6, label='Original')
        ax1.scatter(points_2d_1[~anomaly_mask, 0], points_2d_1[~anomaly_mask,
                    1], c='red', alpha=0.6, label='Reprojected')
        ax1.set_title(f'{camera_names[0]}: Original vs Reprojected (Valid Points)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True)

        # Plot for camera 2
        ax2.scatter(original_2d_2[~anomaly_mask, 0],
                    original_2d_2[~anomaly_mask, 1], c='blue', alpha=0.6, label='Original')
        ax2.scatter(points_2d_2[~anomaly_mask, 0], points_2d_2[~anomaly_mask,
                    1], c='red', alpha=0.6, label='Reprojected')
        ax2.set_title(f'{camera_names[1]}: Original vs Reprojected (Valid Points)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.grid(True)
    else:
        ax1.scatter(original_2d_1[~anomaly_mask, 0],
                    original_2d_1[~anomaly_mask, 1], c='blue', alpha=0.6, label='Original')
        ax1.scatter(points_2d_1[~anomaly_mask, 0], points_2d_1[~anomaly_mask,
                    1], c='red', alpha=0.6, label='Reprojected')
        ax1.set_title(f'{camera_names[0]}: Original vs Reprojected (Valid Points)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True)

    plt.tight_layout()
    plt.show()


def plot_anomaly_vs_original(triangulation_results, stereo='A', camera_names=['Camera 1', 'Camera 2']):
    """
    Plot the original 2D YOLO points vs triangulated 2D points flagged as anomalies
    The idea to be able to see what we are flagging as anomalies
    """
    if len(camera_names) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
    scale_factor = 2
    # Extract points
    original_2d_1 = np.array([[row[f'{stereo}1_x'] * scale_factor, row[f'{stereo}1_y']
                             * scale_factor] for _, row in triangulation_results.iterrows()])
    points_2d_1 = np.array([eval(p)
                           for p in triangulation_results[f'{stereo}1_reprojected_point']])
    original_2d_2 = np.array([[row[f'{stereo}2_x'] * scale_factor, row[f'{stereo}2_y']
                             * scale_factor] for _, row in triangulation_results.iterrows()])
    points_2d_2 = np.array([eval(p)
                           for p in triangulation_results[f'{stereo}2_reprojected_point']])
    anomaly_mask = triangulation_results['anomaly_detected']

    if len(camera_names) > 1:
        # Plot for camera 1
        ax1.scatter(original_2d_1[anomaly_mask, 0],
                    original_2d_1[anomaly_mask, 1], c='blue', alpha=0.6, label='Original')
        ax1.scatter(points_2d_1[anomaly_mask, 0], points_2d_1[anomaly_mask,
                    1], c='red', alpha=0.6, label='Reprojected')
        ax1.set_title(f'{camera_names[0]}: Original vs Reprojected (Anomalies)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True)

        # Plot for camera 2
        ax2.scatter(original_2d_2[anomaly_mask, 0],
                    original_2d_2[anomaly_mask, 1], c='blue', alpha=0.6, label='Original')
        ax2.scatter(points_2d_2[anomaly_mask, 0], points_2d_2[anomaly_mask,
                    1], c='red', alpha=0.6, label='Reprojected')
        ax2.set_title(f'{camera_names[1]}: Original vs Reprojected (Anomalies)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        ax2.grid(True)
    else:
        ax1.scatter(original_2d_1[anomaly_mask, 0],
                    original_2d_1[anomaly_mask, 1], c='blue', alpha=0.6, label='Original')
        ax1.scatter(points_2d_1[anomaly_mask, 0], points_2d_1[anomaly_mask,
                    1], c='red', alpha=0.6, label='Reprojected')
        ax1.set_title(f'{camera_names[0]}: Original vs Reprojected (Anomalies)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True)

    plt.tight_layout()
    plt.show()
