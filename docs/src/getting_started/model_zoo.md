---
title: Model zoo
sidebar_position: 1
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Model Zoo

:::warning  
The model zoo is currently only compatible with the `AiCamera()` device in the Application Module Library.  
:::

The models available in the Application Module Library are sourced from [the Raspberry Pi model zoo library](https://github.com/raspberrypi/imx500-models). These pre-trained models are optimized for use with the IMX500 image sensor.

This library contains a variety of models suitable for different computer vision tasks, such as **image classification**, **object detection**, **segmentation** and **pose estimation**, which can be easily deployed on modlib-compatible devices.

## Examples

Here are some examples demonstrating how to access and utilize models from the Model Zoo for different AI tasks. Each example showcases how to import the appropriate model from the zoo, deploy it on an AiCamera device, and process the incoming frames to perform the specific AI task.

<Tabs>
  <TabItem value="classification" label="Classification" default>

```python title="classifier.py"
import cv2

from modlib.devices import AiCamera
from modlib.models.zoo import EfficientNetB0

device = AiCamera()
model = EfficientNetB0()
device.deploy(model)

with device as stream:
    for frame in stream:

        for i, label in enumerate([model.labels[id] for id in frame.detections.class_id[:3]]):
            text = f"{i+1}. {label}: {frame.detections.confidence[i]:.2f}"
            cv2.putText(frame.image, text, (50, 30 + 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)

        frame.display()
```

  </TabItem>
  <TabItem value="object-detection" label="Object Detection">

```python title="detector.py"
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]
        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]

        annotator.annotate_boxes(frame, detections, labels=labels)
        frame.display()
```

  </TabItem>
  <TabItem value="segmenation" label="Segmentation">

```python title="segment.py"
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import DeepLabV3Plus

device = AiCamera()
model = DeepLabV3Plus()
device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:

        annotator.annotate_segments(frame, frame.detections)
        frame.display()
```

  </TabItem>
  <TabItem value="pose-estimation" label="Pose Estimation">

```python title="posenet.py"
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import Posenet

device = AiCamera()
model = Posenet()
device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:

        annotator.annotate_poses(frame, frame.detections)
        frame.display()
```

  </TabItem>
</Tabs>



## Classification Models

| network_name          | post_processor     | color_format | preserve_aspect_ratio | network                                                                                       |
|-----------------------|--------------------|--------------|-----------------------|-----------------------------------------------------------------------------------------------|
| EfficientNetB0        | pp_cls_softmax     | RGB          | False                 | [imx500_network_efficientnet_bo.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_efficientnet_bo.rpk)      |
| EfficientNetLite0     | pp_cls_softmax     | RGB          | False                 | [imx500_network_efficientnet_lite0.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_efficientnet_lite0.rpk)   |
| EfficientNetV2B0      | pp_cls             | RGB          | True                  | [imx500_network_efficientnetv2_b0.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_efficientnetv2_b0.rpk)    |
| EfficientNetV2B1      | pp_cls             | RGB          | True                  | [imx500_network_efficientnetv2_b1.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_efficientnetv2_b1.rpk)    |
| EfficientNetV2B2      | pp_cls             | RGB          | True                  | [imx500_network_efficientnetv2_b2.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_efficientnetv2_b2.rpk)    |
| MNASNet1_0            | pp_cls_softmax     | RGB          | False                 | [imx500_network_mnasnet1.0.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_mnasnet1.0.rpk)           |
| MobileNetV2           | pp_cls             | RGB          | True                  | [imx500_network_mobilenet_v2.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_mobilenet_v2.rpk)         |
| MobileViTXS           | pp_cls_softmax     | BGR          | True                  | [imx500_network_mobilevit_xs.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_mobilevit_xs.rpk)         |
| MobileViTXXS          | pp_cls_softmax     | RGB          | True                  | [imx500_network_mobilevit_xxs.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_mobilevit_xxs.rpk)        |
| RegNetX002            | pp_cls_softmax     | RGB          | False                 | [imx500_network_regnetx_002.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_regnetx_002.rpk)          |
| RegNetY002            | pp_cls_softmax     | RGB          | False                 | [imx500_network_regnety_002.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_regnety_002.rpk)          |
| RegNetY004            | pp_cls_softmax     | RGB          | False                 | [imx500_network_regnety_004.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_regnety_004.rpk)          |
| ResNet18              | pp_cls_softmax     | RGB          | False                 | [imx500_network_resnet18.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_resnet18.rpk)             |
| ShuffleNetV2X1_5      | pp_cls_softmax     | RGB          | False                 | [imx500_network_shufflenet_v2_x1_5.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_shufflenet_v2_x1_5.rpk)   |
| SqueezeNet1_0         | pp_cls_softmax     | RGB          | False                 | [imx500_network_squeezenet1.0.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_squeezenet1.0.rpk)        |

## Object Detection Models

| network_name                | post_processor           | color_format | preserve_aspect_ratio | network                                                                                       |
|-----------------------------|--------------------------|--------------|-----------------------|-----------------------------------------------------------------------------------------------|
| EfficientDetLite0           | pp_od_efficientdet_lite0 | RGB          | True                  | [imx500_network_efficientdet_lite0_pp.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_efficientdet_lite0_pp.rpk)      |
| NanoDetPlus416x416          | pp_od_bscn               | BGR          | False                 | [imx500_network_nanodet_plus_416x416_pp.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_nanodet_plus_416x416_pp.rpk)    |
| SSDMobileNetV2FPNLite320x320| pp_od_bscn               | RGB          | False                 | [imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk) |

## Segmentation Models

| network_name  | post_processor | color_format | preserve_aspect_ratio | network                                                                                       |
|---------------|----------------|--------------|-----------------------|-----------------------------------------------------------------------------------------------|
| DeepLabV3Plus | pp_segment     | RGB          | False                 | [imx500_network_deeplabv3plus.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_deeplabv3plus.rpk)|

## Pose Estimation Models

| network_name  | post_processor | color_format | preserve_aspect_ratio | network                                                                                       |
|---------------|----------------|--------------|-----------------------|-----------------------------------------------------------------------------------------------|
| Posenet | pp_posenet     | RGB          | False                 | [imx500_network_posenet.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_posenet.rpk)|
| Higherhrnet | pp_higherhrnet     | RGB          | False                 | [imx500_network_higherhrnet_coco.rpk](https://github.com/raspberrypi/imx500-models/raw/main/imx500_network_higherhrnet_coco.rpk)|

<br />
<br />
  