---
title: Recording
sidebar_position: 7
---

# Creating and Playback recordings

Recording and playback functionality is essential when building applications. It allows you to capture specific scenarios and replay them as many times as needed. This capability enables repeated testing and optimization of your application on a given scenario without requiring live input. Note that the AI detection results are stored along with the frames, ensuring consistent testing of application behavior.

Creating and playing back recordings can be used for:
- **Consistent Testing:** Replaying the same scenario ensures reproducible testing conditions.
- **Faster Development Iteration:** No need to re-run the AI model on a physical device every time, speeding up debugging and development.
- **Benchmarking & Performance Analysis:** Compare different algorithm versions against the same recorded data.
- **Offline Testing & Debugging:** Work with recorded data without requiring live camera input.

<br></br>

<div style={{display: 'flex', gap: '20px'}}>
    <div style={{flex: 1}}>
        ![Speed](gifs/recording.gif)
        <div style={{textAlign: 'center', fontSize: '0.7em', marginTop: '-20px'}}>
            **Recording**
        </div>
    </div>
    <div style={{flex: 1}}>
        ![Speed](gifs/playback.gif)
        <div style={{textAlign: 'center', fontSize: '0.7em', marginTop: '-20px'}}>
            **Playback**
        </div>
    </div>
</div>

## Create the recording

The following script demonstrates how to create a recording using `modlib`:
(Press `esc` to end the recording.)

```python title="rec.py"
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import Posenet

from modlib.devices.playback import Recorder, JsonCodec, PickleCodec

device = AiCamera()
model = Posenet()
device.deploy(model)

annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)
rec = Recorder(directory='./temp/recordings', codec=JsonCodec())

with device as stream:
    for frame in stream:
        
        # Save the frame to the recording immediately to ensure it is unaltered.
        rec.add(frame)

        detections = frame.detections[frame.detections.confidence > 0.4]
        annotator.annotate_keypoints(frame, detections)
        frame.display()
```


## Playback the recording

The following script demonstrates how to play back a previously recorded session:
Ensure that you specify the path to the newly created recording file and that the codec used matches the one with which the recording was created (in this case, JsonCodec).

```python title="play.py"
from modlib.apps import Annotator
from modlib.devices.playback import Playback, JsonCodec, PickleCodec

device = Playback(recording="./temp/recordings/recording_2025-03-10_14-12-00.json", codec=JsonCodec())
annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
    for frame in stream:
        print(frame.detections)

        detections = frame.detections[frame.detections.confidence > 0.4]
        annotator.annotate_keypoints(frame, detections)
        frame.display()
```

:::note  
Note that the Playback device does not require any physically connected camera device.  
:::
