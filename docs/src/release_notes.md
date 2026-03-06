---
title: Release Notes
sidebar_position: 6
---


# Release Notes 🚀

## Modlib 1.4.0

### ⭐ **New Features**

- **Data Injection for the AiCamera**: Inject preprocessed input tensors into the IMX500 to evaluate the on-device RPK model. One can choose model preprocessing (best accuracy, comparable with InterpreterClient), model + ISP limitation (no padding, matches camera behavior), or full ISP preprocessing. See [Data Injection](getting_started/data_injection.md) for more information. 
- **Model Evaluators for Classification, Object Detection, Pose Estimation and Segmentation**: Dedicated evaluators (`ClassificationEvaluator`, `COCOEvaluator`, `COCOPoseEvaluator`, `VocSegEvaluator`) compute standard metrics (e.g. top-1/top-5 accuracy, COCO AP/AR, mIoU) from model predictions and ground truth for reproducible benchmarking.
- **InterpreterClient device**: New client device that runs Modlib against an external inference server over HTTP. Use float or quantized ONNX/Keras models on your own hardware while keeping the same device API; ideal for development and comparing server-side vs edge (AiCamera) results.
- **Inference Server Examples**: Example scripts and Docker setup in `examples/interpreters/` show how to run an interpreter server that supports quantized ONNX and Keras models, so you can plug in the InterpreterClient and run evaluations end-to-end.
- **Model preprocessor function for all models in the model zoo**: Every zoo model exposes a `pre_process` fucntion. Use it for data injection and InterpreterClient pipelines.
- **IMX500 ISP utilities** (`modlib.devices.imx500.isp`): New module that collects IMX500 ISP-related helpers for preparing the input tensor for data injection. Includes lower-level functionalities (resize/crop, normalize/quantize, padding, denormalize)
- **Dataset Source**: New `Dataset` source that recursively scans a directory for images and yields `DatasetSample` that are ideal for evaluation workflows.
- **Triton® threading support**: Use Python threading with the Triton® Smart Camera the same way as with the AiCamera, as demonstrated in `examples/triton/threads.py`.


### ❌ **Removed / Replaced**

- **Keras & ONNX Interpreter devices**: The previous in-process Keras and ONNX interpreter devices are removed. Use the new InterpreterClient with a separate inference server (see `examples/interpreters/`) for float or quantized ONNX/Keras models.
- **Removed unused Frame `input_tensor` field**: The placeholder `input_tensor` field on `Frame` has been removed. Use the model’s `pre_process` output or device-specific handling (e.g. AiCamera `enable_input_tensor`) when you need the input tensor or its visualization.


### 📦 **Distribution & Infrastructure**

- **Linter extended to examples and tests**: Ruff linting and formatting now run on the `examples/` and `tests/` directories as well as the main library for consistent style and fewer issues across the repo.


## Modlib 1.3.0

### ⭐ **New Features**

- **AiCamera on Debian Trixie support**: Extended compatibility to support Debian Trixie, enabling users to run AiCamera on the latest Debian stable release.
- **Python 3.11, 3.12, 3.13 support**: Expanded Python version compatibility to include Python 3.11, 3.12, and 3.13. Note: AiCamera still requires the system python version.
- **llms.txt documentation format**: Include the llms.txt format, enabling Large Language Models (LLMs) and AI coding assistants to better understand and work with modlib's API. This includes improved docstrings throughout the codebase for better API reference documentation.
- **Threading example**: New example (`examples/apps/threads.py`) demonstrating how to use Python threading with modlib devices. It shows how to perform background tasks concurrently while processing frames from the device stream.
- **`InstanceSegmentation` and `Segment` result split**: Separated semantic segmentation (`Segments`) and instance segmentation (`InstanceSegments`) into distinct result types. `Segments` provides pixel-level semantic segmentation masks, while `InstanceSegments` includes instance-level masks with bounding boxes, class IDs, and confidence scores.


### 🐛 **Bug Fixes**

- **Fix Triton® Port issue**: Resolved communication port issues with the Triton® Smart Camera device that could cause connection failures or errors during device initialization and model deployment.


### 📦 **Distribution & Infrastructure**

- **PyPI Release**: First official release of modlib on PyPI, making it easier for users to install and manage the library through standard Python package management tools (`pip install`).
- **Improved Docs action workflow**: Improved GitHub Actions workflow for docs generation with automatic versioning and auto `llms.txt` file generation.


## Modlib 1.2.0

### ⭐ **New Features**  

- **AiCamera Device ID**:  
Enhanced the `AiCamera` class with a new method `get_device_id()` to fetch the device ID.
Aditional example `device_id.py` to retrieve the unique device ID from the AI Camera.
- **YOLO Classifier**:  
Introduced the `yolo-classifier.py` example scripts for enhanced AI model demonstrations.
- **YOLO Segmentation**:  
Introduced the `yolo-segment.py` example scripts for enhanced AI model demonstrations.
- **YOLO detection models in model zoo**:  
Added Ultralytics YOLOv8n and YOLO11n models to the model zoo.


### 🐛 **Bug Fixes**

- Fixed `ROI` class with methods for JSON serialization:
  - `json()` to convert the ROI to a JSON-serializable dictionary.
  - `from_json()` to create an ROI from a JSON-serializable dictionary.
- Modified `Frame` class to ensure proper handling of ROI during JSON conversion.


### 📦 **Distribution & Infrastructure**

- Improved unit testing based on recorded scenarios and test cases for better coverage and reliability.


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