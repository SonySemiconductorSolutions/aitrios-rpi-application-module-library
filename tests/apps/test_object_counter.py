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
import numpy as np

from modlib.apps import ObjectCounter, Annotator
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

from tests.test_devices import test_apps_device
from tests.models.test_results_detections import sample_detections


def test_counter_empty():

    counter = ObjectCounter()

    for i in range(10):
        assert counter.get(i) == 0


def test_counter_detections(sample_detections):
    # Case no tracker
    sample_detections = sample_detections.copy()
    sample_detections.tracker_id = None

    counter = ObjectCounter()
    counter.update(sample_detections)

    assert counter.get(0) == 2
    assert counter.get(1) == 1
    assert counter.get(2) == 0

    counter.update(sample_detections)

    assert counter.get(0) == 2 * 2
    assert counter.get(1) == 1 * 2
    assert counter.get(2) == 0 * 2


def test_counter_detections_tracklets(sample_detections):
    # Case with tracklets

    counter = ObjectCounter()
    UPTIME = 20

    for i in range(50):
        counter.update(sample_detections)

        if i <= UPTIME:
            assert counter.get(0) == 0
            assert counter.get(1) == 0
            assert counter.get(2) == 0
        elif i > UPTIME:
            assert counter.get(0) == 2
            assert counter.get(1) == 1
            assert counter.get(2) == 0


def test_object_counter(test_apps_device):

    counter = ObjectCounter()
    annotator = Annotator()
    LABELS = SSDMobileNetV2FPNLite320x320().labels

    with test_apps_device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.50]
            counter.update(detections)
            
            # print(counter.get(0), counter.get(42))

            # labels = [f"#{t} {LABELS[c]}: {s:0.2f}" for _, s, c, t in detections]
            # annotator.annotate_boxes(frame, detections, labels=labels)
            # annotator.set_label(image=frame.image, x=430, y=30, color=(200, 200, 200), label="Total people detected " + str(counter.get(0)))
            # frame.display()

    assert counter.get(0) == 100 # Person
    assert counter.get(42) == 100 # Tennis racket
    assert counter.get(1) == 0 # Something else
        

def test_object_counter_tracker(test_apps_device):

    counter = ObjectCounter()
    annotator = Annotator()
    LABELS = SSDMobileNetV2FPNLite320x320().labels

    with test_apps_device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.50]
            detections.tracker_id = np.array([1, 2]) # Adding tracklets to the static image: Simulating a tracker.update
            counter.update(detections)

            # print(counter.get(0), counter.get(42))

            # labels = [f"#{t} {LABELS[c]}: {s:0.2f}" for _, s, c, t in detections]
            # annotator.annotate_boxes(frame, detections, labels=labels)
            # annotator.set_label(image=frame.image, x=430, y=30, color=(200, 200, 200), label="Total people detected " + str(counter.get(0)))
            # frame.display()
        
    assert counter.get(0) == 1 # Person
    assert counter.get(42) == 1 # Tennis racket
    assert counter.get(1) == 0 # Something else