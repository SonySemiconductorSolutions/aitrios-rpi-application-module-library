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
