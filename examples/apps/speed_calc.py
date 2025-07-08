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

from modlib.apps import Annotator, ColorPalette, BYTETracker
from modlib.apps.calculate import SpeedCalculator
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416


class BYTETrackerArgs:
    track_thresh: float = 0.30
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


model = NanoDetPlus416x416()
device = AiCamera()
device.deploy(model)

distance_per_pixel = 0.00742  # Need to recalibrate when camera is repositioned
tracker = BYTETracker(BYTETrackerArgs())
speed = SpeedCalculator()
annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.50]
        detections = tracker.update(frame, detections)

        # Calculate and retrieve speed per tracked object.
        speed.calculate(frame, detections)
        current_speeds = [speed.get_speed(t, average=False) for t in detections.tracker_id]

        labels = [f"{s * distance_per_pixel * 3.6:0.2f}kph" if s is not None else "..." for s in current_speeds]
        frame.image = annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)
        frame.display()
