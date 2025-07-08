---
title: Motion
sidebar_position: 8
---
import ApiLink from '@site/src/components/ApiLink';

# Motion

The <ApiLink to="/api-reference/apps/motion#motion">Motion</ApiLink> component is designed to identify the change in Motion from frame to frame. It subtracts the current frame from the previous frame to see the change in pixel values. These changes are then turned into bbox results and can be used in combination with other modules.


<div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start' }}>
  <div style={{ flex: '1' }}>
    ![Motion](gifs/motion.gif)
  </div>
  <div style={{ flex: '1' }}>
    ![Motion Matcher](gifs/matcher_motion.gif)
  </div>
</div>

Below an example of how one can use the Motion in the Application Module Library.


```python title="motion.py"
import cv2
from modlib.devices import AiCamera
from modlib.apps import Annotator
from modlib.apps.motion import Motion

device = AiCamera(frame_rate=15)
motion = Motion()
annotator = Annotator(thickness=2)

with device as stream:
    for frame in stream:
        motion_bboxes = motion.detect(frame)
        frame.image = annotator.annotate_boxes(frame,motion_bboxes,skip_label=True)
        frame.display()
```
Motion can be used with other modules like the [Matcher](examples/matcher.md) module, to match motion with Detections produced by any AI model. This provides additional information that could be valueable to applications where movement is involved. 

An example of how to use the Matcher together below:

```python title="motion_matcher.py"
import cv2
from modlib.devices import AiCamera
from modlib.apps import Annotator, Matcher
from modlib.apps.motion import Motion
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

device = AiCamera(frame_rate=15)
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

motion = Motion()
annotator = Annotator(thickness=2)
matcher = Matcher()

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.50]
        motion_bboxes = motion.detect(frame)
        detections = detections[matcher.match(detections, motion_bboxes)]

        labels = [f"# MOVING" for _, s, c, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)
        annotator.annotate_boxes(frame,motion_bboxes,skip_label=True)
        frame.display()

```