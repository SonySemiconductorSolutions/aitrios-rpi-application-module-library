#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv2
import json

from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.devices.frame import Frame
from modlib.models.zoo import (
    EfficientNetB0,
    SSDMobileNetV2FPNLite320x320,
    Posenet,
    DeepLabV3Plus
)


def save_frame(device, model, output_file, N=1):
    
    device.deploy(model)

    frame_list = []

    warmup = 60

    with device as stream:
        for i, frame in enumerate(stream):
            
            # Allow for some warmup lens adjustments
            if i < warmup:
                continue

            frame_list.append(frame.json())

            if len(frame_list) == N:
                break

    # Save the list of frames
    with open(output_file, 'w') as json_file:
        json.dump(frame_list, json_file)


if __name__ == "__main__":

    OUTPUT_FILE = 'tennis_player.json'
    MODEL = SSDMobileNetV2FPNLite320x320()
    
    save_frame(
        device=AiCamera(),
        model=MODEL,
        output_file=OUTPUT_FILE
    )

    # Check the saved frame
    with open(OUTPUT_FILE, 'r') as json_file:
        loaded_frames = json.load(json_file)
    frame = Frame.from_json(loaded_frames[0]) # First frame

    # Annotate & display
    annotator = Annotator()

    # # Classification
    # for i, label in enumerate([MODEL.labels[id] for id in frame.detections.class_id[:3]]):
    #     text = f"{i+1}. {label}: {frame.detections.confidence[i]:.2f}"
    #     cv2.putText(frame.image, text, (50, 30 + 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)

    # # Object detection
    # detections = frame.detections[frame.detections.confidence > 0.20]
    # labels = [f"{MODEL.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
    # annotator.annotate_boxes(frame, detections, labels=labels)

    frame.display()
    
    # Hold frame open until closed
    while True:
        if cv2.waitKey(1) & 0xFF == 27: break # ESC key to exit  
        if cv2.getWindowProperty("Application", cv2.WND_PROP_VISIBLE) < 1: break # Window closed
    cv2.destroyAllWindows()
