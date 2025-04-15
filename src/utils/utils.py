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


def detect_chessboard(frame):
    """
    Detects chessboard corners in an image with sub-pixel accuracy.
    Uses adaptive thresholding and image normalization for robust detection
    across different lighting conditions.
    """
    pattern_size = (9, 6)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    winSize = (5, 5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, winSize, (-1, -1), criteria)

    return ret, corners
