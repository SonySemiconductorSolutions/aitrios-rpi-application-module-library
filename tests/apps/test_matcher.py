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

import os
import pytest

from modlib.apps.matcher import Matcher
from modlib.devices.playback import JsonCodec, PickleCodec, Playback


def test_matcher():

    recording = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/recordings/tea_nobg.json"
    device = Playback(recording, codec=JsonCodec())

    matcher = Matcher(min_overlap_threshold=0.01, hysteresis=0)

    MATCH = False

    with device as stream:
        for frame in stream:
            # frame.image = np.zeros((frame.height, frame.width, 3), dtype=np.uint8)

            detections = frame.detections
            p = detections[detections.class_id == 0]  # Person
            c = detections[detections.class_id == 41]  # Cup

            MATCH = matcher.match(p, c)
            
            # # NOTE: Manual check the matcher output
            # print(MATCH)

    
    assert MATCH == True