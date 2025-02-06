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
import cv2
import pytest

from modlib.devices import Images, Video

from tests.utils import get_imagenet_samples, get_coco_samples, get_tracking_video


@pytest.fixture
def imagenet_source():
    
    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/assets/imagenet_samples"
    _ = get_imagenet_samples(current_dir)

    return Images(current_dir)


@pytest.fixture
def coco_source():

    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/assets/coco_samples"
    _ = get_coco_samples(current_dir)

    return Images(current_dir)


@pytest.fixture
def tracking_video():
    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/assets"
    r = get_tracking_video(current_dir)

    return Video(r['path'])


def test_imagenet_source(imagenet_source):
    assert len(imagenet_source.image_files) == 3

    _ = imagenet_source.get_frame() # goldfish
    _ = imagenet_source.get_frame() # ipod
    _ = imagenet_source.get_frame() # daisy
    assert imagenet_source.get_frame() is None


def test_coco_source(coco_source):
    assert len(coco_source.image_files) == 3

    _ = coco_source.get_frame() # stop sign (724)
    _ = coco_source.get_frame() # skier (785)
    _ = coco_source.get_frame() # tenis player (885)
    assert coco_source.get_frame() is None


def test_tracking_video(tracking_video):
    assert tracking_video.total_frames == 329
    assert tracking_video.frame_number == 0

    _ = tracking_video.get_frame()
    assert tracking_video.frame_number == 1

    _ = tracking_video.get_frame()
    assert tracking_video.frame_number == 2

    # ...
    tracking_video.frame_number = 329 - 1

    _ = tracking_video.get_frame()
    assert tracking_video.frame_number == 329

    assert tracking_video.get_frame() is None
    