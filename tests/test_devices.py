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
from modlib.devices.playback import JsonCodec, PickleCodec, Playback


class RepeatingMockPlayback(Playback):
    def __init__(
            self,
            recording,
            codec=JsonCodec(),
            repeat_frame_index=0,
            repeat_count=100,
            headless=True,
            timeout=None,
        ):
        
        super().__init__(recording, codec, headless, timeout)
        
        self.repeat_frame_index = repeat_frame_index
        self.repeat_count = repeat_count
        self.current_count = 0

        self._frame_to_repeat = self._load_frame(repeat_frame_index)

    def _load_frame(self, index: int) -> Frame:
        self.file.seek(0)
        for i in range(index + 1):
            frame = self.codec.decode(self.file)
            if frame is None:
                raise IndexError(f"Frame index {index} out of range in recording.")
        return frame
    
    def __iter__(self):
        self.current_count = 0
        return self

    def __next__(self):
        self.check_timeout()

        if self.current_count >= self.repeat_count:
            raise StopIteration

        self.current_count += 1
        return self._frame_to_repeat



@pytest.fixture
def test_classifier_device():
    recording = f"{os.path.dirname(os.path.abspath(__file__))}/assets/recordings/cls_samples.json"
    return Playback(recording, codec=JsonCodec())


@pytest.fixture
def test_detector_device():
    recording = f"{os.path.dirname(os.path.abspath(__file__))}/assets/recordings/od_samples.json"
    return Playback(recording, codec=JsonCodec())


# @pytest.fixture
# def test_segmentation_device():
#     recording = f"{os.path.dirname(os.path.abspath(__file__))}/assets/recordings/segmentation_recording.json"
#     return RecordingMockDevice(recording)


@pytest.fixture
def test_apps_device():
    # Device returning the skier for N frames
    recording = f"{os.path.dirname(os.path.abspath(__file__))}/assets/recordings/od_samples.json"
    IDX_TENNIS_PLAYER = 2
    return RepeatingMockPlayback(recording, repeat_frame_index=IDX_TENNIS_PLAYER, repeat_count=100)


def test_capture_length(test_classifier_device):

    count = 0
    with test_classifier_device as stream:
        for _ in stream:
            count += 1

    assert count == 3 # Expected number of frames in the recording


def test_capture_length_2(test_detector_device):

    count = 0
    with test_detector_device as stream:
        for _ in stream:
            count += 1

    assert count == 3 # Expected number of frames in the recording


def test_capture_repeating(test_apps_device):

    count = 0
    with  test_apps_device as stream:
        for frame in stream:
            count += 1

    assert count == 100 # Expected number of frames in the repeating mock playback


def test_classifier(test_classifier_device):
    
    expected_classes = [
        113,       # (a_snail.png): 'snail'
        549,       # (b_envelope.jpg): 'envelope'
        985        # (c_daisy.jpg): 'daisy'
    ]
    
    with test_classifier_device as stream:
        for i, frame in enumerate(stream):
            assert frame.detections.class_id[0] == expected_classes[i], f"Detected class {frame.detections.class_id[0]} does not match expected class {expected_classes[i]}"


def test_detector(test_detector_device):
    
    # Each entry is a dict: {class_id: expected_count, ...}
    expected_class_counts = [
        {11: 1},          # (724.jpg): 1 stop sign
        {0: 1, 30: 1},          # (785.jpg): 1 person, 1 skis
        {0: 1, 38: 1, 32: 1}    # (885.jpg): 1 person, 1 tennis racket, 1 sports ball
    ]
    
    with test_detector_device as stream:
        for i, frame in enumerate(stream):

            class_ids = frame.detections.class_id.tolist()
            expected = expected_class_counts[i]
            actual = {cid: class_ids.count(cid) for cid in set(class_ids)}
            assert actual == expected, f"Frame {i}: Expected {expected}, got {actual}"
