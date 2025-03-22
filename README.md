## Main scripts 
`select_frames_calibration.py` -> to get the frames used for intrinsic calibration for each camera. It process each video and select frames where the chessboard is visible

`select_stereo_frames.py` -> get the frames for stereo calibration, same as the script above but for the common view of each pair of cameras

`calibrate_intrinsic.py` -> go through all the frames for each camera to compute the intrinsic parameters and store them in `output/intrinsic.json`