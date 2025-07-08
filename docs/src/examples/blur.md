---
title: Blur
sidebar_position: 6
---
import ApiLink from '@site/src/components/ApiLink';

# Blurring

The blurring functionality in modLib provides tools for privacy and object anonymization in video streams. You can selectively blur faces or entire objects (like people) in real-time.

<div style={{display: 'flex', gap: '20px'}}>
    <div style={{flex: 1}}>
        ![Blur Face](gifs/blur-face.gif)
        <div style={{textAlign: 'center', fontSize: '0.7em', marginTop: '-20px'}}>
            **Bluring Faces**
        </div>
    </div>
    <div style={{flex: 1}}>
        ![Blur Object](gifs/blur-object.gif)
        <div style={{textAlign: 'center', fontSize: '0.7em', marginTop: '-20px'}}>
            **Bluring Objects**
        </div>
    </div>
</div>

## Blur Faces

The <ApiLink to="/api-reference/apps/blur#blur_face">Blur Face</ApiLink> function allows to blur the face of a person detected by drawing a boundry box using the keypoints of the face and then blurring the region inside the boundary box.

Below a full example of how to use blur_face in the Application Module Library.

```python title="blur_face.py"
from modlib.apps import Annotator, blur
from modlib.devices import AiCamera
from modlib.models.zoo import Posenet

device = AiCamera()
model = Posenet()
device.deploy(model)

threshold = 0.1

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > threshold]
        # get the face keypoint_scores and check if their average is greater than threshold
        filter = [sum(d[:5]) / 5 > threshold for d in detections.keypoint_scores]
        filtered_detections = detections[filter]

        blur.blur_face(frame, filtered_detections, padding=7)
        frame.display()
```

## Blur Objects

The <ApiLink to="/api-reference/apps/blur#blur_object">Blur Object</ApiLink> function allows to blur a person detected and then blurring the region inside the boundary box.

Below a full example of how to use blur_object in the Application Module Library.

```python title="blur_object.py"
from modlib.apps import Annotator, blur
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]
        detections = detections[detections.class_id == 0]  # Person

        blur.blur_object(frame, detections)

        labels = [f"#{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)

        frame.display()

```