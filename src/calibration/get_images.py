"""
Open two webcams and save images from them to use for stereo calibration
If not checboard, then use: https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures?resource=download
"""

import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

num = 0 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break


    frame = cv2.resize(frame, (640, 480))

    # Simulate stereo by splitting into left and right images
    width = frame.shape[1] // 2
    imgL = frame[:, :width]  # Left half
    imgR = frame[:, width:]  # Right half

    # Display images
    cv2.imshow("Stereo Left", imgL)
    cv2.imshow("Stereo Right", imgR)

    # Key event handling
    k = cv2.waitKey(5)

    if k == 27:  # ESC to exit
        break
    elif k == ord('s'):  # Save images when 's' is pressed
        cv2.imwrite(f'images/stereoLeft/imageL{num}.png', imgL)
        cv2.imwrite(f'images/stereoRight/imageR{num}.png', imgR)
        
        print(f"Images saved! imageL{num}.png and imageR{num}.png")
        num += 1

cap.release()
cv2.destroyAllWindows()
