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

        labels = ["# MOVING" for _, s, c, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)
        annotator.annotate_boxes(frame, motion_bboxes, skip_label=True)
        frame.display()
