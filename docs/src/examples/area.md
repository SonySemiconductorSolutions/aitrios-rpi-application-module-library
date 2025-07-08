---
title: Area
sidebar_position: 1
---
import ApiLink from '@site/src/components/ApiLink';

# Area

The <ApiLink to="/api-reference/apps/area#area">Area</ApiLink> component allows you to define a polygonal region within a frame by specifying a list of at least three points. Each point is defined as a pair of **normalized** coordinates `[x, y]`, where x and y are in the range `[0, 1]`, relative to the frame dimensions. 

The `area.contains(detections)` method returns a boolean mask indicating whether the center of each detection's bounding box is inside the defined polygon.

<div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start' }}>
  <div style={{ flex: '1' }}>
    The Area component enables more detailed analysis. Using the obtained mask one can:
    - **Filter** all the detections to only include those within the specified area.
    - **Count** the number of detections that fall inisde the area.
  </div>
  <div style={{ flex: '1' }}>
    ![Area](gifs/area.gif)
  </div>
</div>

Below a full example of how to use Area's in the Application Module Library.

```python title="area.py"
from modlib.apps import Annotator, Area
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

areas = [
    Area(points=[[0.1, 0.1], [0.1, 0.9], [0.4, 0.9], [0.4, 0.1]]),  # area 1
    Area(points=[[0.6, 0.2], [0.8, 0.1], [0.9, 0.4], [0.8, 0.8], [0.6, 0.7]]),  # area 2
]

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]
        detections = detections[detections.class_id == 0]  # Person

        for area in areas:
            d = detections[area.contains(detections)]
            color = (0, 255, 0) if len(d) > 0 else (0, 0, 255)
            annotator.annotate_area(frame, area=area, color=color, label=f"Count: {len(d)}")

        labels = [f"#{model.labels[c]}: {s:0.2f}" for _, s, c, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)
        frame.display()
```