import cv2
import numpy as np

# Chessboard parameters
rows = 7  # Inner corners
cols = 10
square_size = 60  # mm per square
dpi = 300  # High resolution

# Create the chessboard pattern
chessboard = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            chessboard[
                i * square_size : (i + 1) * square_size,
                j * square_size : (j + 1) * square_size,
            ] = 255

# Save high-res image
cv2.imwrite("chessboard.png", chessboard)
