#
# Copyright 2025 Sony Semiconductor Solutions Corp. All rights reserved.
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

import time
import threading

from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320


def background_task(device: AiCamera):
    while device._running:
        frame = device.get_frame()

        # simulate work
        time.sleep(1)

        print(f'Performing task: {frame.timestamp}')


device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)
thread = threading.Thread(target=background_task, args=(device,))


with device as stream:
    thread.start()
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]

        labels = [f"{model.labels[c]}: {s:0.2f}" for _, s, c, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)

        frame.display()

# Wait for the thread to finish
thread.join()