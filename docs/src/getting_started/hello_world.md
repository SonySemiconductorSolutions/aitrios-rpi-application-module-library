---
title: Hello world
sidebar_position: 0
---


# Getting Started

## Setup your device

:::warning  
This getting started guide is focussing on the **Raspberry Pi AI Camera**. If you are working with the **Triton® Smart camera**, please proceed immediately to the [Triton® documentation](../devices/triton).
:::

The Raspberry Pi AI Camera is an extremely capable piece of hardware, enabling you to build powerful AI applications on your Raspberry Pi. By offloading the AI inference to the IMX500 accelerator chip, more computational resources are available to handle application logic right on the edge!

If you haven't done so already, make sure to [verify](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) that your AI Camera is set up correctly.

## Install the Application Module Library

Install the Application Module Library in your current Python (virtual) environment.  

1. Ensure that your Raspberry Pi runs the latest software:

```shell
sudo apt update && sudo apt full-upgrade
sudo apt install imx500-all
```

2. (Optional) Create and enable a virtual environment.

```shell
python -m venv .venv
source .venv/bin/activate
```

3. One can use pip to install the library in your project Python environment.
```shell
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git
```

:::warning  
We are currently working on publishing Modlib to the PyPI index. This will make it easier for developers to install and manage the library using pip and allow for a more streamlined setup process. Stay tuned for updates!
:::

## Verify your setup

Let's verify that our camera is connected and the Application Module Library is installed correctly.  
Create a new Python file named `hello_world.py`. and run the following code to see a camera preview.

```python title="hello_world.py"
from modlib.devices import AiCamera

device = AiCamera()

with device as stream:
    for frame in stream:
        frame.display()
```

## First example

Let's dive into our first example! This example will demonstrate how to use the AI Camera to detect objects in real-time using a pre-trained SSDMobileNet model.

First, extend the `hello_world.py` example in your project directory and add the following code:

```python title="hello_world.py"
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

A brief overview of the key steps in this example:
- Initiate the `AiCamera`device
- Initiate the pre-packaged `SSDMobileNetV2FPNLite320x320` model from the Zoo.
- Deploy the model to the device.
- Start the stream and visualize the detections that have a confidence greater then the given threshold (0.55).

