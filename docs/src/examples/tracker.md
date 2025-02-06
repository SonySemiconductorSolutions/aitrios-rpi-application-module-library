---
title: Tracker
sidebar_position: 0
---

# ByteTrack

An implementation of ByteTrack ([github](https://github.com/ifzhang/ByteTrack)) in the Application Module Library.

ByteTrack provides a state of the art method for multi-object tracking by associating all detection boxes, including low-confidence ones. Unlike traditional methods that discard uncertain detections, ByteTrack recovers true objects and ensures continuous tracking, even in challenging conditions like occlusion or partial visibility.

The [BYTETracker](../api-reference/apps/tracker/#bytetracker) offers a simple, effective, and robust solution for real-world tracking needs. Below a full example on how to use the BYTETracker in the Application Module Library.

```python title="tracker.py"
from modlib.apps import Annotator, BYTETracker
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
annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]
        detections = detections[detections.class_id == 0]  # Person

        detections = tracker.update(frame, detections)

        labels = [f"#{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)

        frame.display()
```