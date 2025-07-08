---
title: Devices
sidebar_position: 3
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ApiLink from '@site/src/components/ApiLink';

# Devices

The Application Module Library provides an abstraction layer to work with different devices. In general all devices use a common API:
```python
from modlib.devices import AiCamera, Triton

device = AiCamera()
# (optionally) device.deploy(model)

with device as stream:
    for frame in stream:
        frame.display()
```

Where <ApiLink to="/api-reference/devices/frame#frame">Frame</ApiLink> does contain all the needed information to build your application or verify your model output results.

## Supported Devices

- **Raspberry Pi AI Camera**: [https://www.raspberrypi.com/products/ai-camera/](https://www.raspberrypi.com/products/ai-camera/)  
- **Triton® Smart (IMX501)**: [https://thinklucid.com/product/triton-smart-12-3-mp-imx501/](https://thinklucid.com/product/triton-smart-12-3-mp-imx501/)

## Device Features

|                                   |                       | AiCamera | Triton® |
|-----------------------------------|-----------------------|----------|-------|
| Mode                              | Headless              | ✅       | ✅    |
|                                   | Input Tensor          | ✅       | ✅    |
| Adjustable frame rate             |                       | ✅       | ✅    |
| Adjustable sensor image size      | Auto selected binning | ✅       | ✅    |
| Multi - device                    |                       | ✅       | WIP   |
| Input tensor cropping             |                       | ✅ (runtime)  | ✅ (at init) |
| Image cropping                    |                       | ✅ (at init)  | ✅ (at init) |
| Deploy                            | Packaged Models       | RPK package   | FPK package (sensor specific) |
|                                   | Converted Models      | ✅       | WIP   |
|                                   | ONNX (PyTorch) Models | ✅       | WIP   |
|                                   | KERAS (Tensorflow) Models | ✅       | WIP   |
| FPS, DPS, RPS - threaded          |                       | ✅       | ✅    |
| Model Zoo Compatibility           |                       | ✅       | ❌    |
| Application Modules Compatibility |                       | ✅       | ✅    |
