---
title: Heatmap
sidebar_position: 4
---
import ApiLink from '@site/src/components/ApiLink';

# Heatmap

The <ApiLink to="/api-reference/apps/heatmap#heatmap">Heatmap</ApiLink> component is used to visualize the density of detected objects across time by overlaying a heatmap onto the frame. The `heatmap.update(frame, detections)` method stores detections across multiple frames to track the frequency of object appearances in different regions of the frame.

The heatmap is created by dividing the frame into a grid of cells, with each cell accumulating detection counts. The `cell_size` parameter controls the size of each grid cell.

![Heatmap](gifs/heatmap.gif)

Below an example of how to use a Heatmap in the Application Module Library.


```python title="heatmap.py"
from modlib.apps import Annotator, Heatmap
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

heatmap = Heatmap(cell_size=50)
annotator = Annotator()

with device as stream:
    for frame in stream:

        detections = frame.detections[frame.detections.class_id == 0]  # Person
        detections = detections[detections.confidence > 0.55]

        # Create Heatmap
        heatmap.update(frame, detections)

        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)
        frame.display()
```