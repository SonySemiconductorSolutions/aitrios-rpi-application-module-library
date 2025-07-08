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

from modlib.apps import Annotator, Area
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)

areas = [
    Area(points=[[0.1, 0.1], [0.1, 0.9], [0.4, 0.9], [0.4, 0.1]]),  # area 1
    Area(points=[[0.6, 0.2], [0.8, 0.1], [0.9, 0.4], [0.8, 0.8], [0.6, 0.7]]),  # area 2
]

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]
        detections = detections[detections.class_id == 0]  # Person

        for area in areas:
            d = detections[area.contains(detections)]
            color = (0, 255, 0) if len(d) > 0 else (0, 0, 255)
            annotator.annotate_area(frame, area=area, color=color, label=f"Count: {len(d)}", alpha=0.2)

        labels = [f"#{model.labels[c]}: {s:0.2f}" for _, s, c, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.2)
        frame.display()
