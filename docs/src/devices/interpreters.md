---
title: Interpreters
sidebar_position: 2
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ApiLink from '@site/src/components/ApiLink';

# Interpreter Devices

Interpreter devices allow you to develop and test your application locally on your development PC without requiring a physical camera. This is particularly useful for rapid prototyping, debugging, and testing your AI models and application logic before deploying to a physical device. With interpreter devices, you can use your own image data source and develop your application as if it were connected to a camera image sensor. 

## Keras Interpreter

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
        frame.display(resize_image=True)
        time.sleep(1)
```

  </TabItem>
  <TabItem value="video" label="Video" default>

```python
from modlib.devices import KerasInterpreter, Video

device = KerasInterpreter(source=Video("./path/to/video.mp4"))

with device as stream:
    for frame in stream:
        frame.display()
```

  </TabItem>
</Tabs>

3. Deploy and run your model 

The Keras Interpreter device requires to deploy a custom model with `model_type=MODEL_TYPE.KERAS`. 
For more information on how to deploy custom models see [> Custom Models](../getting_started/custom_models.md).

:::info  
Note that the pre-processing method in the Model object is required when working with interpreter devices.  
:::


## ONNX Interpreter

1. Install additional interpreter runtime dependencies.

```
pip install onnxruntime
```

2. Define your data source

You can decide what data to work with. Choos between `Images` or a `Video`.

<Tabs>
  <TabItem value="images" label="Images" default>

```python
import time

from modlib.devices import Images, ONNXInterpreter

device = ONNXInterpreter(source=Images("./path/to/image/directory"))

with device as stream:
    for frame in stream:
        frame.display(resize_image=True)
        time.sleep(1)
```

  </TabItem>
  <TabItem value="video" label="Video" default>

```python
from modlib.devices import ONNXInterpreter, Video

device = ONNXInterpreter(source=Video("./path/to/video.mp4"))

with device as stream:
    for frame in stream:
        frame.display()
```

  </TabItem>
</Tabs>

3. Deploy and run your model 

The ONNX Interpreter device requires to deploy a custom model with `model_type=MODEL_TYPE.ONNX`. 
For more information on how to deploy custom models see [> Custom Models](../getting_started/custom_models.md).

:::info  
Note that the pre-processing method in the Model object is required when working with interpreter devices.  
:::