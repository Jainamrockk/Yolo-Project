
# YOLO Car Counting Project

## Overview

The YOLO Car Counting Project is designed to detect and count vehicles in real-time using advanced computer vision techniques. By leveraging the YOLO (You Only Look Once) model for object detection, this project efficiently tracks cars, buses, and motorbikes in a video stream, providing accurate counts as they cross a defined area.

## Features

- **Real-Time Vehicle Detection:** Utilizes the YOLOv8 model for fast and precise identification of vehicles in various conditions.
- **Robust Tracking:** Implements the SORT (Simple Online and Realtime Tracking) algorithm to maintain consistent tracking of vehicle identities.
- **Custom Masking:** Applies a mask to focus detection on relevant areas, improving accuracy and reducing false positives.
- **Counting Logic:** Counts vehicles that cross a predefined line in the video frame, providing an up-to-date tally of passing vehicles.

## Installation

### Prerequisites

Make sure you have Python 3.x installed. Install the necessary packages with:

```bash
pip install opencv-python cvzone numpy ultralytics sort
```

### YOLO Weights

Download the YOLOv8 weights from the [Ultralytics repository](https://github.com/ultralytics/yolov8/releases) and place them in the `Yolo-Weights` directory.

## Usage

1. Place your video file (e.g., `cars.mp4`) and mask image (e.g., `mask.jpg`) in the project directory.
2. Run the main script to start processing the video.
3. The application will open a window displaying the video feed with detected vehicles and the total count of vehicles passing through the designated limits.

## Results

The system visualizes the detection and counting process by overlaying bounding boxes around detected vehicles, assigning unique IDs for tracking, and displaying the count of vehicles that have crossed the defined line in real-time.


## Acknowledgments

- **Ultralytics YOLO:** For providing the YOLO model, which is foundational to this project.
- **OpenCV:** For powerful image processing capabilities.
- **cvzone:** For simplifying common OpenCV tasks.

---
