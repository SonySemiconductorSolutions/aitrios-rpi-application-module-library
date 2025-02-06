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

from modlib.apps.annotate import ColorPalette, Annotator
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

from tests.test_devices import test_apps_device


def test_annotations(test_apps_device):

    annotator = Annotator(color=ColorPalette.default(), thickness=2, text_thickness=1, text_scale=0.5)
    LABELS = SSDMobileNetV2FPNLite320x320().labels

    with test_apps_device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.55]
            
            labels = [f"{LABELS[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
            annotator.annotate_boxes(frame, detections, labels=labels)

            # NOTE: manual check
            # frame.display()
