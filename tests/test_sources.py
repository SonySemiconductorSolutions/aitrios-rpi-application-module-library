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
from pathlib import Path

from modlib.devices import Images, Video, Dataset

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


@pytest.fixture
def imagenet_dataset():

    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/assets/imagenet_samples"
    _ = get_imagenet_samples(current_dir)

    return Dataset(current_dir)


def test_imagenet_source(imagenet_source):
    assert imagenet_source.timestamp is not None
    assert len(imagenet_source) == 3

    _ = imagenet_source.get_frame() # goldfish
    _ = imagenet_source.get_frame() # ipod
    _ = imagenet_source.get_frame() # daisy
    assert imagenet_source.get_frame() is None


def test_imagenet_source_iterator(imagenet_source):
    for i, _ in enumerate(imagenet_source):
        if i > 2:
            raise ValueError(f"Unexpected number of images: {i}")


def test_coco_source(coco_source):
    assert len(coco_source) == 4

    _ = coco_source.get_frame() # stop sign (724)
    _ = coco_source.get_frame() # skier (785)
    _ = coco_source.get_frame() # tenis player (885)
    _ = coco_source.get_frame() # surfboard (1490)
    assert coco_source.get_frame() is None


def test_tracking_video(tracking_video):
    assert tracking_video.timestamp is not None
    assert len(tracking_video) == 329
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


def test_imagenet_dataset(imagenet_dataset):
    assert imagenet_dataset.timestamp is not None
    assert len(imagenet_dataset) == 3

    _ = imagenet_dataset.get_frame() # goldfish
    _ = imagenet_dataset.get_frame() # ipod
    _ = imagenet_dataset.get_frame() # daisy

    assert imagenet_dataset.get_frame() is None


def test_imagenet_dataset_iterator(imagenet_dataset):
    for i, sample in enumerate(imagenet_dataset):
        if i == 0:
            assert sample.image_id == "a_goldfish"
        elif i == 1:
            assert sample.image_id == "b_ipod"
        elif i == 2:
            assert sample.image_id == "c_daisy"
        else:
            raise ValueError(f"Unexpected number of images: {i}")


def test_non_exist():
    with pytest.raises(FileNotFoundError):
        Images(Path("path/to/non_exist"))
    with pytest.raises(FileNotFoundError):
        Video(Path("path/to/non_exist"))
    with pytest.raises(FileNotFoundError):
        Dataset(Path("path/to/non_exist"))
