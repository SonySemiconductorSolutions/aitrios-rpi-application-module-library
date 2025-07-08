---
title: Counter
sidebar_position: 2
---
import ApiLink from '@site/src/components/ApiLink';

# Counter

The <ApiLink to="/api-reference/apps/object_counter#objectcounter">ObjectCounter</ApiLink> component is designed to work together with a tracker to keep a persistent count of detected objects over time. By using the tracklet information, it is able to distinguish between different bounding boxes across frames. When a new tracklet is detected, it increments the count for that tracklet's associated class.  

- The `counter.update(detections)` method processes the current frame's detections and updates the count for each tracked object of any class_id.
- The `counter.get(class_id)` method returns the total count of objects for a specific `class_id` detected across frames.

![Counter](gifs/counter.gif)

An example of how the Object Counter can be used in the Application Module Library below:

```python title="counter.py"
from modlib.apps import Annotator, BYTETracker, ObjectCounter
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

tracker = BYTETracker(BYTETrackerArgs())
people_counter = ObjectCounter()
annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]
        detections = detections[detections.class_id == 0]  # Person
        detections = tracker.update(frame, detections)

        people_counter.update(detections)
        annotator.set_label(
            image=frame.image,
            x=430,
            y=30,
            color=(200, 200, 200),
            label="Total people detected " + str(people_counter.get(0)),
        )

        labels = [f"#{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]
        annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)

        frame.display()
```