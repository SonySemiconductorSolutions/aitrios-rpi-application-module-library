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

import pytest
from modlib.apps import blur
from tests.test_devices import test_apps_device

def test_blur_object(test_apps_device):
    with test_apps_device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.55]
            detections = detections[detections.class_id == 0]  # Person

            blur.blur_object(frame, detections, intensity=50)

            # NOTE: manual check
            # frame.display()

def test_blur_face(test_apps_device):
    pass
    # TODO: add posnet to test_apps 

            