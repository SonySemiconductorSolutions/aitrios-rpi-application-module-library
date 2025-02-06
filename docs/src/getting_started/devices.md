---
title: Devices
sidebar_position: 3
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Devices

The Application Module Library provides an abstraction layer to work with different devices. In general all devices use a common API:
```python
from modlib.devices import AiCamera

device = AiCamera()
# (optionally) device.deploy(model)

with device as stream:
    for frame in stream:
        frame.display()
```

Where [Frame](../api-reference/devices/frame#frame) does contain all the needed information to build your application or verify your model output results.

## AI Camera

Compatible with all Raspberry Pi computers, the [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/) takes advantage of Sony’s IMX500 Intelligent Vision Sensor to help you create impressive vision AI applications and neural network models using the on-module AI processor.

## Interpreter devices

Interpreter devices allow you to develop and test your application locally on your development PC without requiring a physical camera. This is particularly useful for rapid prototyping, debugging, and testing your AI models and application logic before deploying to a physical device. With interpreter devices, you can use your own image data source and develop your application as if it were connected to a camera image sensor. 

### Keras Interpreter

1. Install additional interpreter runtime dependencies.

```
pip install tensorflow
```

2. Define your data source

You can decide what data to work with. Choos between `Images` or a `Video`.

<Tabs>
  <TabItem value="images" label="Images" default>

```python
import time

from modlib.devices import Images, KerasInterpreter

device = KerasInterpreter(source=Images("./path/to/image/directory"))

with device as stream:
    for frame in stream:
        frame.display(resize=True)
        time.sleep(1)
```

  </TabItem>
  <TabItem value="video" label="Video" default>

```python
from modlib.devices import KerasInterpreter, Video

device = KerasInterpreter(source=Video("./path/to/video.mp4"))

with device as stream:
    for frame in stream:
        frame.display(resize=False)
```

  </TabItem>
</Tabs>

3. Deploy and run your model 

:::info  
Note that the pre-processing method in the Model object is required when working with interpreter devices.  
:::
