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

from modlib.apps.area import Area

from tests.test_devices import test_apps_device


def test_area(test_apps_device):

    area = Area([[0.1,0.1], [0.1,0.9], [0.9,0.9], [0.9,0.1]]) #Area covers whole camera view 

    AREA = False

    with test_apps_device as stream:
        for frame in stream:
            
            detections = frame.detections[frame.detections.confidence > 0.50]
            detections = detections[area.contains(detections)]

            if len(detections) > 0:
                AREA = True

            # NOTE: Manual check the area output
            # print(AREA)
    
    assert AREA == True