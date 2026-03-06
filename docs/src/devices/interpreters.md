---
title: Interpreters
sidebar_position: 2
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ApiLink from '@site/src/components/ApiLink';

# Interpreter Devices

Interpreter devices allow you to develop and test your application locally on your development PC without requiring a physical camera. This is particularly useful for rapid prototyping, debugging, and testing your AI models and application logic before deploying to a physical device. With interpreter devices, you can use your own image data source and develop your application as if it were connected to a camera image sensor.

Additionally, interpreter devices allow you to obtain a solid baseline for model evaluation. The evaluation results from interpreter devices can serve as a benchmark when assessing the performance of your models after they have been quantized, packaged, and converted for deployment on the IMX500 AiCamera, enabling you to measure any accuracy trade-offs introduced during the model training and optimization process.  

Interpreter devices are designed to work on a `server`-`client` basis. Where the user is fully responsible for setting up the server, and Modlib integrates a `InterpreterClient` that mimics the API's of any other device in the Application Module Library.

## Interpreter Server 

The Interpreter Server can be designed and run on any hardware of you choice. The only requirement is that an HTTP endpoint is available to the `InterpreterClient` and integrates the following methods:

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| `POST` | `/init` | Initializes a new inference session and loads the model. | Accepts any JSON object (dict). The client sends the `data` parameter from `device.deploy(model, data={...})` as JSON. | Must return JSON with a `session_id` field (string). Optionally may include `status: "error"` and `error` (string) fields to indicate failure.<br/><br/>**Example:**<br/>```{"session_id": "abc123...","status": "success"}``` |
| `DELETE` | `/session/{session_id}` | Closes and cleans up an inference session. | Path parameter: `session_id` (string) - The session identifier returned from `/init` | Must return HTTP 200 OK. Response body is not parsed by the client. |
| `GET` | `/input_tensor_size/{session_id}` | Retrieves the expected input tensor dimensions for the model. | Path parameter: `session_id` (string) - The session identifier | Must return JSON with an `input_tensor_size` field (array). The client accesses `input_tensor_size[0]` and `input_tensor_size[1]` for width and height respectively. Optionally may include `status: "error"` and `error` (string) fields to indicate failure.<br/><br/>**Example:**<br/>```{"input_tensor_size": [320, 320]}``` |
| `POST` | `/infer/{session_id}` | Performs inference on the provided input tensor. | Path parameter: `session_id` (string) - The session identifier<br/><br/>Multipart form data with a file field named `input_npy` containing a NumPy array in `.npy` format (binary). | Must return binary content in `.npz` format (NumPy compressed archive). The client loads this as `np.load()` and extracts tensors from it. Must return HTTP 200 OK on success. |


For an example implenation of such a Interpreter Server, please have a look at the the example folder in Modlib: https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library/tree/main/examples/interpreters


## Interpreter Client

The `InterpreterClient` works like any other device in Modlib. 

:::note  
The InterpreterClient requires an inference server running simultaneously.
:::

```
from modlib.apps import Annotator
from modlib.devices import Video, InterpreterClient
from modlib.models.zoo import YOLO11n

device = InterpreterClient(
    source=Video("./examples/assets/palace.mp4"),
    endpoint="http://localhost:8000",
    enable_input_tensor=False,
)

model = YOLO11n()
device.deploy(model, data={
    "model_uri": f"/models/yolo11n_imx_model/yolo11n_imx.onnx",
    "options": {"is_quantized": True},
})

annotator = Annotator()

with device as stream:
    for frame in stream:

        detections = frame.detections[frame.detections.confidence > 0.40]
        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)

        frame.display()
```