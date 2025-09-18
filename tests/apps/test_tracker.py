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
import importlib.resources
import numpy as np

from modlib.apps import BYTETracker
from modlib.apps.annotate import ColorPalette, Annotator

from tests.test_devices import test_apps_device


def test_annotations(test_apps_device):

    LABELS = np.genfromtxt(
        str(importlib.resources.files("modlib.models.zoo") / "assets" / "coco_labels_80.txt"),
        dtype=str,
        delimiter="\n"
    )

    class BYTETrackerArgs:
        track_thresh: float = 0.25
        track_buffer: int = 30
        match_thresh: float = 0.8
        aspect_ratio_thresh: float = 3.0
        min_box_area: float = 1.0
        mot20: bool = False

    tracker = BYTETracker(BYTETrackerArgs())
    annotator = Annotator(color=ColorPalette.default(), thickness=2, text_thickness=1, text_scale=0.5)
    
    test_results = set()

    with test_apps_device as stream:
        for frame in stream:
            detections = tracker.update(frame, frame.detections)
            test_results.update(detections.tracker_id)

            # # NOTE: Manually check the tracklets
            # labels = [f"#{t} {LABELS[c]}: {s:0.2f}" for _, s, c, t in detections]
            # annotator.annotate_boxes(frame, detections, labels=labels)
            # frame.display()

    # Exactly 3 tracklets, 1 person, 1 tennis racket and 1 sports ball
    assert len(test_results) == 3 and all(x in test_results for x in (1, 2, 3))
