# Real-Time Hand Tracking with Virtual Boundary Warning

## Overview
This project is a **proof-of-concept (POC)** for real-time hand tracking using a standard webcam feed. The system detects when a user's hand approaches a virtual boundary on the screen and triggers a visual warning when the hand reaches critical proximity.  

The POC demonstrates **real-time fingertip tracking** using classical computer vision techniques (no MediaPipe, OpenPose, or cloud AI APIs are used).  

## Features
- **Hand/Fingertip tracking** using color segmentation, contours, convex hull, and background subtraction.
- **Virtual boundary detection**: A rectangle is drawn on the screen, and the system computes the hand’s distance to this boundary.
- **Distance-based state logic**:
  - `SAFE` – Hand comfortably far from the boundary.
  - `WARNING` – Hand approaching the boundary.
  - `DANGER` – Hand very close or touching the boundary.
- **Visual feedback overlay**:
  - Current state displayed in real-time.
  - “DANGER DANGER” warning when in danger state.
  - Distance from the virtual boundary shown for debugging.
- **Real-time performance**: Target ≥ 8 FPS on CPU-only execution.
- **User controls**:
  - `c` – Calibrate skin color by placing hand in the calibration box.
  - `b` – Toggle background subtraction on/off.
  - `r` – Reset calibration.
  - `q` – Quit the application.
## How It Works

1. **Video capture**  
   Accesses the webcam feed at 640x480 resolution.

2. **Color segmentation**  
   Converts frames to YCrCb and HSV color spaces to detect skin regions.

3. **Background subtraction** (optional)  
   Helps remove static background noise.

4. **Contour detection**  
   Finds hand contours using OpenCV.

5. **Fingertip detection**  
   Identifies the fingertip as the point farthest from the contour’s center.

6. **Distance computation**  
   Calculates the distance of the fingertip from a virtual rectangle at the center of the screen.

7. **State classification**  
   - Compares distance to predefined warning and danger thresholds.  
   - Updates overlay state and triggers “DANGER DANGER” warning when required.

8. **Visual overlays**  
   Shows the hand contour, convex hull, fingertip, virtual rectangle, and mask.
   

## Usage Tips

- Place your hand inside the **calibration box (top-left corner)** and press `c` to calibrate skin color.
- Toggle **background subtraction** using `b` to improve detection in noisy backgrounds.
- Reset calibration with `r` if the hand is not being detected properly.
- The system works best under **good lighting conditions** with minimal clutter in the background.
