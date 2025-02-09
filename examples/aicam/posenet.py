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

from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models.zoo import Posenet

device = AiCamera()
model = Posenet()
device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.5]
        annotator.annotate_poses(frame, detections)
        frame.display()
