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

from modlib.devices import AiCamera
from modlib.models.zoo import EfficientNetB0

device = AiCamera()
model = EfficientNetB0()
device.deploy(model)

with device as stream:
    for frame in stream:

        for i, label in enumerate([model.labels[id] for id in frame.detections.class_id[:3]]):
            text = f"{i+1}. {label}: {frame.detections.confidence[i]:.2f}"
            cv2.putText(frame.image, text, (50, 30 + 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)

        frame.display()
