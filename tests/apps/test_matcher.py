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

from modlib.apps.matcher import Matcher

from tests.test_devices import test_apps_device


def test_matcher(test_apps_device):

    matcher = Matcher(min_overlap_threshold=0.01, hysteresis=0)

    MATCH = False

    with test_apps_device as stream:
        for frame in stream:
            
            detections = frame.detections[frame.detections.confidence > 0.50]
            p = detections[detections.class_id == 0]  # Person
            s = detections[detections.class_id == 42] # Tennis Racket

            MATCH = matcher.match(p, s)

            # NOTE: Manual check the matcher output
            # print(MATCH)
    
    assert MATCH == True