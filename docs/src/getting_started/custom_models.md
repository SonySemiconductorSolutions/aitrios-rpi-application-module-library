---
title: Custom models
sidebar_position: 2
---
import ApiLink from '@site/src/components/ApiLink';


# Custom models


The Application Module Library provides a method to deploy custom-trained models to devices using a similar API to deploying models from the Model Zoo. Due to its modular design, any custom model will work with the already available devices.

```python
device = AiCamera()
model = CustomModel()
device.deploy(model)
```

This means you can easily adapt your custom models, work with various devices, without significant changes to your application code. This approach ensures consistency in your development process, whether you're using models from the zoo or your own custom-trained models.


## Example

Here's a full example of how you can use custom models with the Application Module Library:

```python
import numpy as np
from typing import List

from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_bscn
from modlib.models.results import Detections


class SSDMobileNetV2FPNLite320x320(Model):
    def __init__(self):
        super().__init__(
            model_file="./path/to/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

        # Optionally define self.labels

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_bscn(output_tensors)
    

device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:
        # NOTE: frame.detections contains the result returned by model.post_process()
        detections = frame.detections[frame.detections.confidence > 0.55]
        annotator.annotate_boxes(frame, detections, alpha=0.3, corner_radius=10)
        frame.display()
```


## Specification

To take advantage of modlib's model abstraction layer one needs to follow a certain set of rules.
The first one is to initialize the inherited base <ApiLink to="/api-reference/models/model">Model</ApiLink> class.

**Arguments:**
- model_file (Path): The path to the model file.
- model_type (<ApiLink to="/api-reference/models/model#model_type">MODEL_TYPE</ApiLink>): The type of the model.
- color_format (<ApiLink to="/api-reference/models/model#color_format">COLOR_FORMAT</ApiLink>, optional): The color format of the model (RGB or BGR). Defaults to `COLOR_FORMAT.RGB`.
- preserve_aspect_ratio (bool, optional): Setting the sensor whether or not to preserve aspect ratio of the input tensor. Defaults to `True`.

:::info
Next to RPK_PACKAGED models, one can also provide a CONVERTED, or quantized KERAS/ONNX models.

| model_type              | Expected model_file  | `device.deploy(model)` Functionality         |
|-------------------------|----------------------|----------------------------------------------|
|`MODEL_TYPE.RPK_PACKAGED`|`*.rpk`-file          |Uploads packaged model to device.             |
|`MODEL_TYPE.CONVERTED`   |`packerOut.zip`-file  |Packages model for device & uploads.          |
|`MODEL_TYPE.KERAS`       |`*.keras`-file        |Converts quantized model, packages & uploads. |
|`MODEL_TYPE.ONNX`        |`*.onnx`-file         |Converts quantized model, packages & uploads. |
:::

## Post Processing

Implement the necessary post-processing method, which has a **strictly defined signature and expected output**. 
- **Argument:** output_tensors (`List[np.ndarray]`) A list of output tensors returned by your custom model
- **Returns:** One of the <ApiLink to="/api-reference/models/results#result">Result</ApiLink> types (`Classifications`, `Detections`, `Poses`, `Segments` or `Anomaly`)

```python
def post_process(self, output_tensors: List[np.ndarray]) -> Union[Classifications, Detections, Poses, Segments, Anomaly]:
```

For convenience, the most common post-processing functions are included in the <ApiLink to="/api-reference/models/post_processors">post_processing library</ApiLink> of the Application Module Library. As a rule: **The output of the post-processor function will be available in the `frame.detections` variable during runtime.**

```python
device = AiCamera()
model = CustomModel()  # Defines the model.post_process() function
device.deploy(model)

with device as stream:
    for frame in stream:
        print(frame.detections) # Contains the result of the model.post_process() function
```

## Pre Processing (Optional)

The pre-processing method is only required when deploying the model to an <u>Interpreter Device</u> or when using <u>Data-Injection</u>.
Similar to the post-processing, pre-processing has a predefined function signature and expected output.
- **Argument:** image (`np.ndarray`) The input image of the Source to be processed.
- **Returns:** (`Tuple[np.ndarray, np.ndarray]`) A tuple (input_tensor_image, input_tensor):
    - Preprocessed input tensor image as a NumPy array.
    - Input tensor ready for model inference.

```python
def pre_process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
```
