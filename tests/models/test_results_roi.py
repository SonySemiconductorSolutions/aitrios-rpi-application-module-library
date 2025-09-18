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

import json
import pytest

from modlib.models import ROI


def test_roi_basic_access_and_iteration():
    roi = ROI(left=0.1, top=0.2, width=0.3, height=0.4)

    # attribute access
    assert roi.left == 0.1
    assert roi.top == 0.2
    assert roi.width == 0.3
    assert roi.height == 0.4

    # item access
    assert roi[0] == 0.1
    assert roi[1] == 0.2
    assert roi[2] == 0.3
    assert roi[3] == 0.4

    # iteration order
    assert list(iter(roi)) == [0.1, 0.2, 0.3, 0.4]


def test_roi_json_serializable_roundtrip():
    roi = ROI(left=10, top=20, width=30, height=40)

    # Ensure json() output is json.dumps-able
    json_dict = roi.json()
    dumped = json.dumps(json_dict)
    assert isinstance(dumped, str)

    # Roundtrip
    restored = ROI.from_json(json.loads(dumped))
    assert restored.left == pytest.approx(roi.left)
    assert restored.top == pytest.approx(roi.top)
    assert restored.width == pytest.approx(roi.width)
    assert restored.height == pytest.approx(roi.height)

