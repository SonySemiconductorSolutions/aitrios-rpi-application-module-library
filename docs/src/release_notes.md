---
title: Release Notes
sidebar_position: 5
---


# Release Notes 🚀

## Modlib 1.1.0

### 📸 **NEW DEVICE**

**Triton® Smart Camera**  
Full support for Triton® camera with Sony IMX501.
 

### ⭐ **New Features**  

- **Object Blurring Module**  
Application module allowing bounding box objects and faces (when using pose models) to be blurred.  
Methods: `blur_object`, `blur_face`.  
- **Distance Calculation Module**:  
Calculates pixel distance between objects. Can be scaled to real-world units
- **Speed Calculation Module**:  
Calculates the speed of tracked objects over time based on pixel distance changes.
- **Frame Cropping**:  
Region of interest (ROI) cropping added to the frame.display method.
- **New Pre- and Post-processing Functions**
    - HigherHRNet model (post-processor)
    - (Raw) YOLO models (post-processor); object-, keypoint- & segmentation
    - YOLO Ultralytics exports (post-processors); object- & keypoint-detection
    - PersonLab keypoint detection (C++ post-processor) with improved performance for gauge demo.
    - YOLO (pre-processor)
- **Anomaly Density Calculation Utility**:  
A new utility for analyzing anomaly density in visual data.
- **Motion Detection Module**:    
Detects motion via frame differencing and returns bounding boxes.
- **Instance Segmentation Support**:  
Converts segmentation results into instance segmentations with oriented bounding boxes.
- **RPS vs DPS Performance Script**:  
Plots and analysis to find optimal Frame Rate for a model.
- **Caching, Recording & Playback Module**:  
Enables scenario recording and replay for development and debugging.
- **DPS Performance Monitoring**:  
Alerts on significant mismatch between RPS (requested) and DPS (delivered) frame rates with recommendations.
- **Improved Annotator Visuals**:  
Enhancements for bounding boxes, segments, poses, and anomaly overlays.
- **AiCamera; frame rate and image size**:  
Supports configurable frame rate and high-res inputs with automatic binning for optimal FPS.
- **ONNX Interpeter device**:  
Supports running raw onnx models in modlib with onnx runtime
- **ROI compenatation for every result type**:  
Implements comensate for introduced ROI (input tensor + high res) for Detections/Poses/Segement result types.


### 🐛 **Bug Fixes**

- Fixed issue where the first image in Source: Images mode appeared black.


### ❌ **Removed / Replaced**

- **PiCamera2 depenency**:  
AiCamera device reimplemented without picmera2 as requirement
- **Build System Overhaul**:  
Switched from Poetry to Astral uv with Meson build system.


### 📦 **Distribution & Infrastructure**

- **Library Wheel Distribution**:  
    `cibuildwheel` setup for PyPI with support for:  
    - manylinux 2_35 (x86, aarch64)
    - Windows (amd64)
- 📚 **Documentation Improvements**:  
    - Modlib docs enhanced with better visuals. 
    - Preparation for publishing docs.
- 🧹 **Code Quality**:  Adopted ruff for linting and formatting.