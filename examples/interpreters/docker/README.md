# Interpreter in Docker

This folder contains a small FastAPI-based Keras/TensorFlow and ONNX interpreter packaged for Docker.

**Files:**
- `Dockerfile` — builds the container image, installs Python deps and starts a Uvicorn server on port 8000.
- `server.py` — FastAPI server that exposes endpoints to initialize a model, get input_tensor_size, run inference, and close sessions (`/init`, `/input_tensor_size/{session_id}`, `/infer/{session_id}`, `/session/{session_id}`).
- `interpreter_keras.py` — TensorFlow/Keras model interpreter that loads Keras models (supports MCT quantized models), checks version compatibility, and provides inference functionality.
- `interpreter_onnx.py` — ONNX model interpreter that loads ONNX models using onnxruntime (supports MCT quantized models) and provides inference functionality. 
- `requirements-keras-tf2.13.txt` — Python packages for TensorFlow 2.13 (for object detection models).
- `requirements-keras-tf2.14.txt` — Python packages for TensorFlow 2.14 (for semantic segmentation models like DeepLabV3Plus).
- `requirements-onnx.txt` — Python packages for ONNX runtime models.

## Build the Inference Server containers

**Build for Keras with TensorFlow 2.13**
```bash
docker build --build-arg BACKEND=keras-tf2.13 -t keras-infer:tf2.13 --network=host .
```

**Build for Keras with TensorFlow 2.14**
```bash
docker build --build-arg BACKEND=keras-tf2.14 -t keras-infer:tf2.14 --network=host .
```

**Build for ONNX**
```bash
docker build --build-arg BACKEND=onnx -t onnx-infer:latest --network=host .
```

## Prepare the models

It is necessary that you collect one or a series of quantized models you want to run in the server. A typical model folder structure looks like this. However you have freedom to choose any structure you prefer.
```
/models
├── model-a/
│   └── model-a.keras  # requires keras-tf2.13
├── model-b/
│   └── model-b.keras  # requires keras-tf2.14
├── model-c/
│   └── model-c.onnx   # requires onnx
|   └── ...
...
```

**For example:**  
Running this [Ultralytics YOLO11 tutorial](https://docs.ultralytics.com/integrations/sony-imx500/) will provide the quantized `onnx` models requied to run the interpreter, as well as the IMX500 converted `packerOut.zip` models ready to deploy on the Raspberry Pi AiCamera. By running the provided scripts for all model different AI tasks one would end up with a folder structure like this. Which can be used when running the provided Interpreter server.

```
/models
├── yolo11n_imx_model  # Object Detection
│   ├── dnnParams.xml
│   ├── labels.txt
│   ├── packerOut.zip
│   ├── yolo11n_imx.onnx
│   ├── yolo11n_imx_MemoryReport.json
│   └── yolo11n_imx.pbtxt
├── yolo11n-pose_imx_model  # Pose Estimation
|   └── ...
├── yolo11n-cls_imx_model  # Classification
|   └── ...
├── yolo11n-seg_imx_model  # Instance Segmentation
|   └── ...
...
```

## Run the server as container

Run the container exposing port 8000 and mount a local `models` directory (read-only) into the container at `/models` so the server can access model files:

Example:
```bash
MODELS=<path/to/models/directory>
docker run --rm -p 8000:8000 -v $MODELS:/models:ro onnx-infer:latest
```

