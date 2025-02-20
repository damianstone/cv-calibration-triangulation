import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import sys

# imread_img = cv.imread(image_path)
def print_image(imread_img):
  img = cv.cvtColor(imread_img, cv.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  

def get_frame_size(image_path_pattern):
    image_list = glob.glob(image_path_pattern)
    if not image_list:
        raise FileNotFoundError("No images found")
    img = cv.imread(image_list[0])
    height, width = img.shape[:2] 
    return (width, height)
