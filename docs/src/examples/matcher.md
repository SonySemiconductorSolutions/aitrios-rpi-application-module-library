---
title: Matcher
sidebar_position: 3
---
import ApiLink from '@site/src/components/ApiLink';

# Matcher

The <ApiLink to="/api-reference/apps/matcher#matcher">Matcher</ApiLink> component is designed to identify relationships between two sets of detections based on their spatial overlap. It takes two sets of detections and compares their bounding boxes to determine overlaping area. The `matcher.match(set1, set2)` method returns a boolean mask indicating which detections in the first set overlap with any detection in the second set.

<div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start' }}>
  <div style={{ flex: '1' }}>
    The Matcher can be used to identify context-specific relationships, such as:
    - Determining if a person is holding an object (e.g., a cup).
    - Verifying if a person is wearing a specific item (e.g., a helmet or safety vest).
  </div>
  <div style={{ flex: '1' }}>
    ![Matcher](gifs/matcher.gif)
  </div>
</div>

Below an example of how one can use the Matcher in the Application Module Library.


```python title="matcher.py"
from modlib.apps import Annotator, Matcher
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

matcher = Matcher()
annotator = Annotator()

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.50]

        p = detections[detections.class_id == 0]  # Person
        c = detections[detections.class_id == 46]  # Cup

        detections = p[matcher.match(p, c)]

        labels = [f"# PERSON & CUP" for _, s, c, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)
        frame.display()
```


The Matcher achieves improved performance when used in combination with a tracker, such as [BYTETracker](examples/tracker.md). 

Tracklets, which are unique identifiers assigned to tracked objects across frames, provide an additional piece of information to maintain the correlation between the two sets of detections over time. This makes sure that objects are consistently identified across frames, reducing false matches caused by momentary overlaps or shifting detections.

An example of how to use the Matcher and Tracker together below:

```python title="tracker_matcher.py"
from modlib.apps import Annotator, BYTETracker, Matcher
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

matcher = Matcher()
tracker = BYTETracker(BYTETrackerArgs())
annotator = Annotator()

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.50]
        detections = tracker.update(frame, detections)

        p = detections[detections.class_id == 0]  # Person
        c = detections[detections.class_id == 46]  # Cup

        detections = p[matcher.match(p, c)]

        labels = [f"#{t} PERSON & CUP" for _, s, c, t in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)
        frame.display()
```