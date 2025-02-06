---
title: Custom models
sidebar_position: 2
---


# Custom models


The Application Module Library provides a method to deploy custom-trained models to devices using a similar API to deploying models from the Model Zoo. Due to its modular design, any custom model will work with the already available devices.

```python
device = AiCamera()
model = CustomModel()
device.deploy(model)
```

This means you can easily adapt your custom models, work with various devices, without significant changes to your application code. This approach ensures consistency in your development process, whether you're using models from the zoo or your own custom-trained models.


## Example

Here's a brief example of how you can use custom models with the Application Module Library:

```python
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_bscn

class SSDMobileNetV2FPNLite320x320(Model):
    def __init__(self):
        super().__init__(
            model_file="./path/to/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )

        # Optionally define self.labels

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_bscn(output_tensors)
```


## Specification

To take advantage of modlib's model abstraction layer one needs to follow a certain set of rules.
The first one is to initialize the inherited base [Model](../api-reference/models/model) class.

**Arguments:**
- model_file (Path): The path to the model file.
- model_type ([MODEL_TYPE](../api-reference/models/model#model_type)): The type of the model.
- color_format ([COLOR_FORMAT](../api-reference/models/model#color_format), optional): The color format of the model (RGB or BGR). Defaults to `COLOR_FORMAT.RGB`.
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

Implement the necessary post-processing method, which has a strictly defined signature and expected output. 
- **Argument:** output_tensors (`List[np.ndarray]`) A list of output tensors returned by your custom model
- **Returns:** One of the [Result](../api-reference/models/results#result) types (`Classifications`, `Detections`, `Poses`, `Segments` or `Anomaly`)

```python
def post_process(self, output_tensors: List[np.ndarray]) -> Union[Classifications, Detections, Poses, Segments, Anomaly]:
```

For convenience, the most common post-processing functions are included in the [post_processing library](../api-reference/models/post_processors) of the Application Module Library.

## Pre Processing (Optional)

The pre-processing method is only required when deploying the model to an <u>Interpreter Device</u> or when using <u>Data-Injection</u>.
Similar to the post-processing, pre-processing has a predifined function signature and expected output.
- **Argument:** image (`np.ndarray`) The input image of the Source to be processed.
- **Returns:** (`Tuple[np.ndarray, np.ndarray]`) A tuple (input_tensor_image, input_tensor):
    - Preprocessed input tensor image as a NumPy array.
    - Input tensor ready for model inference.

```python
def pre_process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
```
