## What to ignore

All the folders inside `src/` (e.g. `src/get-intrinsic-params/` ) as they are just notebooks for testing with public data and for learning about calibration

Other files to ignore
- coordinates.py
- trajectory.py
- blender_shot.py

## Start

```bash
python3 -m venv venv-tp
```

```bash
source venv-tp/bin/activate
```

```bash
pip install -r requirements.txt
```

## Main scripts

`select_frames_calibration.py` -> to get the frames used for intrinsic calibration for each camera. It process each video and select frames where the chessboard is visible

`select_stereo_frames.py` -> get the frames for stereo calibration, same as the script above but for the common view of each pair of cameras

`calibrate_intrinsic.py` -> go through all the frames for each camera to compute the intrinsic parameters and store them in `output/intrinsic.json`

`calibrate_stereo_pair.py` -> go through synchronised frames calculating the extrinsic parameters per stereo and rectification

`triangulation.py` -> go through YOLO output calculating the 3D ball position and projecting the calculated ball in the corresponding frame. saving the images in `reprojected_frames/`

**None of the scripts receive any arguments**

For changing inputs or outputs, each of them have something like this at the end of the file:

```python
if __name__ == "__main__":
    root = find_project_root()
    # cam 1
    camera_a = "CAM_A_RIGHT"
    record_name = "2_stereo_view"
    left_video_path = f"{root}/videos/{camera_a}/{record_name}.MOV"

    # cam 2
    camera_b = "CAM_B_RIGHT"
    record_name = "2_stereo_view"
    right_video_path = f"{root}/videos/{camera_b}/{record_name}.MOV"

    # output
    output_path = f"{root}/images/full-stereo/right"

    # pattern variables
    chessboard_size = (9, 6)
    square_size_mm = 60

    process_stereo_videos(
        left_video_path=left_video_path,
        right_video_path=right_video_path,
        record_name=record_name,
        output_dir=output_path,
        pattern_size=chessboard_size,
        num_frames=100,
        skip_seconds=1
    )

```

### Fremes calibration folders
From images/
```bash
STEREOS/
    STEREO_A/
        stereo_frames/
            0000/
                cam_1.png
                cam_2.png
            0001/
            0002/
        filtered_stereo_frames/
            0001/
            0002/
        CAMERA_1/
            intrinsic_frames/
            filtered_intrinsic_frames/
        CAMERA_2/
            intrinsic_frames/
            filtered_intrinsic_frames/
    STEREO_B/
        stereo_frames/
            0000/
                cam_2.png
                cam_3.png
            0001/
            0002/
        filtered_stereo_frames/
            0001/
            0002/
        CAMERA_2/
            intrinsic_frames/
            filtered_intrinsic_frames/
        CAMERA_3/
            intrinsic_frames/
            filtered_intrinsic_frames/
```

### Video calibration folders
From videos/
```bash
STEREOS/
    STEREO_A/
        1_camera_intrinsic.MOV
        1_camera_extrinsic.MOV
        2_camera_intrinsic.MOV
        2_camera_extrinsic.MOV
    STEREO_B/
        3_camera_intrinsic.MOV
        3_camera_extrinsic.MOV
        4_camera_intrinsic.MOV
        4_camera_extrinsic.MOV
```