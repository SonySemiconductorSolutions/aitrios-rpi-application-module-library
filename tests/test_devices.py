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
import json
import time
import pytest

from modlib.devices.frame import Frame


class RecordingMockDevice:
    def __init__(self, recording_json):
        with open(recording_json, "r") as f:
            self.source = json.load(f)

    def __enter__(self):
        self.index = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.source):
            frame_dict = self.source[self.index]
            self.index += 1
            return Frame.from_json(frame_dict)
        else:
            raise StopIteration


class RepeatingMockDevice:
    def __init__(self, frame_dict, N):
        self.frame = Frame.from_json(frame_dict)
        self.N = N

        self.original_image = self.frame.image

    def __enter__(self):
        self.index = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.N:
            self.index += 1
            
            # Make sure to refresh the image for a possible overwritten
            # NOTE: Consider introducing a frame.copy()
            self.frame.image = self.original_image.copy()
            return self.frame
        else:
            raise StopIteration


@pytest.fixture
def test_classifier_device():
    recording = f"{os.path.dirname(os.path.abspath(__file__))}/assets/recordings/classification_recording.json"
    return RecordingMockDevice(recording)


@pytest.fixture
def test_detector_device():
    recording = f"{os.path.dirname(os.path.abspath(__file__))}/assets/recordings/detection_recording.json"
    return RecordingMockDevice(recording)


# @pytest.fixture
# def test_segmentation_device():
#     recording = f"{os.path.dirname(os.path.abspath(__file__))}/assets/recordings/segmentation_recording.json"
#     return RecordingMockDevice(recording)


@pytest.fixture
def test_apps_device():
    # Device returning the skier for N frames
    recording = f"{os.path.dirname(os.path.abspath(__file__))}/assets/recordings/detection_recording.json"
    with open(recording, "r") as f:
        frame_dict_skier = json.load(f)[2]

    return RepeatingMockDevice(frame_dict_skier, N=100)


def test_capture(test_classifier_device):

    count = 0
    with test_classifier_device as stream:
        for _ in stream:
            count += 1

    assert count == len(test_classifier_device.source)


def test_capture_repeating(test_apps_device):

    count = 0
    with  test_apps_device as stream:
        for _ in stream:
            count += 1

    assert count == 100


def test_classifier(test_classifier_device):
    
    expected_classes = [
        1,         # (a_goldfish.jpg): 'goldfish, Carassius auratus'
        605,       # (b_ipod.jpg): 'iPod'
        985        # (c_daisy.jpg): 'daisy'
    ]
    
    with test_classifier_device as stream:
        for i, frame in enumerate(stream):
            assert frame.detections.class_id[0] == expected_classes[i], f"Detected class {frame.detections.class_id[0]} does not match expected class {expected_classes[i]}"


def test_detector(test_detector_device):
    
    expected_classes = [
        [12],    # (724.jpg): stop_sign # NOTE missing ground thruth: car, truck 
        [0],     # (785.jpg): person # NOTE: missing ground thruth ski's
        [0, 42]  # (885.jpg): person, tennis racket
    ]
    
    with test_detector_device as stream:
        for i, frame in enumerate(stream):
            detections = frame.detections[frame.detections.confidence > 0.50]
            assert len(detections.class_id) == len(expected_classes[i]), f"Expected {len(expected_classes[i])} detections, but got {detections.class_id}"
            assert (detections.class_id == expected_classes[i]).all(), f"Class IDs do not match. Expected {expected_classes[i]}, but got {detections.class_id}"

