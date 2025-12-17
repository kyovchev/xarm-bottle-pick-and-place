# Bottle Detection with Robotic Pick and Place

This repository contains the complete algortihm for bottle pick and place operation.

The used robot is xArm7 and the AI model for bottle detection is YOLO Oriented Bounding Boxes. The camera which is used is M5Stack Timer Camera X with RTSP streaming firmware. Any other camera can also be used.

The repository contains the code needed for dataset generation and model training as well as pick and place algorithm and also a simple xArm7 emulator of the XArmAPI which can be used for testing purposes without real robot.

## Requirements

The code is tested on Python 3.10 and Python 3.12. The following packages `requirements.txt` are required.

Make sure that PyTorch is installed and can access the GPU correctly. You can check that everything work as expected with the Jupyter Notebook located in [./utils/bottle_detector.ipynb](./utils/bottle_detector.ipynb).

## Dataset Generation

The script in `utils/capture.py` will allow you to easily capture multiple images which will then be used for the dataset.

After you capture the images you need to label them by using a tool like Roboflow, which supports the YOLO OBB format (Object Detection, OBB/Rotation). You need to label two classes `bottle` and `cap` which correspond to each bottle and each bottle cap position which will be later used for detecting the bottle orientation.

You need to configure the URL of the RTSP Stream in the file `config/detect_obb.json`. You can use the `config/detect_obb.json.tpl` file for the first config file creation since the config file is not tracked by the repo.

## Model Training and Object Detection

Afer the dataset is generated you can train the model with `train.py`.

You need to configure the ArUco tag numbers, locations and the pick and place locations in the file `config/detect_obb.json`. You can use the `config/detect_obb.json.tpl` file for the first config file creation since the config file is not tracked by the repo.

## REST API App for Bottle Detection

This repository provides a REST API App for execution of bottle detection and pick and place coordinates generation.

This is the current and maintained approach. In order to start the app you can use the [run.sh](./run.sh) script or you can run the uvicorn server by execution of the following command:

`uvicorn main:app --host 0.0.0.0 --port 22001`.

This will start the app at port `22001`.

After starting the app you can access the API documentation at the following URL: http://localhost:22001/docs

The documentation will provide you with all the information about the API and the Swager UI will allow you test and experiment with the different queries.

## Stand-alone OpenCV App

This repository provides a helper stand-alone OpenCV app which allows you to se the result of the detection algorithm and to send pick and place command to a helper script for control of the xArm7 robot. It was developed for initial testing and validation purposes and is not maintained since 14.12.2025.

For the execution you need to start from the `utils` folder the `xarm_pick_and_place.py` together with `detect_obb.py`. When you click on the bottle, the pick and place command will be send to the robot.

In the `config/xarm_pick_and_place.json` you need to set the IP address of the robot and whether to run the script in XArmAPI emulation mode. You can use the `config/xarm_pick_and_place.json.tpl` file for the first config file creation since the config file is not tracked by the repo.

## Other Utilities

There are several helper scripts in the `utils` folder:

- `bottle_detector.ipynb` demonstrates the object detection and recognition in a Jyputer Notebook.
- `json_config.py` loads JSON configuration file.
- `manual_detect.py` demonstrates the coordinates calculation with the help of AruCo tags.
- `pick-station-cad` is a folder containing OpenSCAD and STLs useful for a 3D printed bottle despencer and place stations.
- `xarm_emulator.py` is the XArmAPI emulator.
