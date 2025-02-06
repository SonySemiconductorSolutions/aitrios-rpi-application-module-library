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

import numpy as np
import pytest

from modlib.models import Detections


@pytest.fixture
def sample_detections():
    d = Detections()
    d.bbox = np.array([
        [50, 50, 100, 100],   # Bounding box 1: class 0, tracker 1, confidence 0.6, area 2500, width 50, height 50
        [200, 200, 450, 500], # Bounding box 2: class 1, tracker 2, confidence 0.8, area 75000, width 250, height 300
        [600, 100, 650, 130]  # Bounding box 3: class 0, tracker 3, confidence 0.4, area 1500, width 50, height 30
    ])
    d.confidence = np.array([0.6, 0.8, 0.4])
    d.class_id = np.array([0, 1, 0])
    d.tracker_id = np.array([1, 2, 3])
    return d


def test_init(sample_detections):
    assert len(sample_detections.bbox) == 3
    assert len(sample_detections.confidence) == 3
    assert len(sample_detections.class_id) == 3
    assert len(sample_detections.tracker_id) == 3


def test_len(sample_detections):
    assert len(sample_detections) == 3


def test_iter(sample_detections):
    for i, (bbox, confidence, class_id, tracker_id) in enumerate(sample_detections):
        assert np.array_equal(bbox, sample_detections.bbox[i])
        assert confidence == sample_detections.confidence[i]
        assert class_id == sample_detections.class_id[i]
        assert tracker_id == sample_detections.tracker_id[i]


def test_add(sample_detections):
    d1 = Detections()
    d1.bbox = np.array([[0, 0, 10, 10]])
    d1.confidence = np.array([0.9])
    d1.class_id = np.array([1])
    d1.tracker_id = np.array([1])

    d = sample_detections + d1
    assert np.array_equal(d.bbox, np.vstack((sample_detections.bbox, d1.bbox)))
    assert np.array_equal(d.confidence, np.concatenate([sample_detections.confidence, d1.confidence]))
    assert np.array_equal(d.class_id, np.concatenate([sample_detections.class_id, d1.class_id]))
    assert np.array_equal(d.tracker_id, np.concatenate([sample_detections.tracker_id, d1.tracker_id]))


def test_add_type_error():
    d = Detections()
    with pytest.raises(TypeError):
        d = d + 1


def test_getitem_int(sample_detections):
    d = sample_detections[1]
    assert len(d) == 1
    assert np.array_equal(d.bbox[0], sample_detections.bbox[1])
    assert d.confidence[0] == sample_detections.confidence[1]
    assert d.class_id[0] == sample_detections.class_id[1]
    assert d.tracker_id[0] == sample_detections.tracker_id[1]


def test_getitem_slice(sample_detections):
    d = sample_detections[0:2]
    assert len(d) == 2
    assert np.array_equal(d.bbox, sample_detections.bbox[0:2])
    assert np.array_equal(d.confidence, sample_detections.confidence[0:2])
    assert np.array_equal(d.class_id, sample_detections.class_id[0:2])
    assert np.array_equal(d.tracker_id, sample_detections.tracker_id[0:2])


def test_getitem_list(sample_detections):
    indices = [0, 2]
    d = sample_detections[indices]
    assert len(d) == 2
    assert np.array_equal(d.bbox, sample_detections.bbox[indices])
    assert np.array_equal(d.confidence, sample_detections.confidence[indices])
    assert np.array_equal(d.class_id, sample_detections.class_id[indices])
    assert np.array_equal(d.tracker_id, sample_detections.tracker_id[indices])


# Representations
def test_str(sample_detections):
    expected_str = f"Detections(class_id:\t {sample_detections.class_id}, \tconfidence:\t {sample_detections.confidence}, \tbbox_shape: {sample_detections.bbox.shape}, \ttrack_ids:\t {sample_detections.tracker_id})"
    assert str(sample_detections) == expected_str


def test_json(sample_detections):
    json_dict = sample_detections.json()
    assert json_dict["bbox"] == sample_detections.bbox.tolist()
    assert json_dict["confidence"] == sample_detections.confidence.tolist()
    assert json_dict["class_id"] == sample_detections.class_id.tolist()
    assert json_dict["tracker_id"] == sample_detections.tracker_id.tolist()


# Filtering
def test_filter_by_id(sample_detections):
    filtered_detections = sample_detections[sample_detections.class_id == 0]
    assert len(filtered_detections) == 2
    assert np.all(filtered_detections.class_id == 0)


def test_filter_by_set(sample_detections):
    selected_tracks = [1, 3]
    filtered_detections = sample_detections[np.isin(sample_detections.tracker_id, selected_tracks)]
    assert len(filtered_detections) == 2
    assert np.all(np.isin(filtered_detections.tracker_id, selected_tracks))


def test_filter_by_confidence(sample_detections):
    filtered_detections = sample_detections[sample_detections.confidence > 0.5]
    assert len(filtered_detections) == 2


def test_filter_by_bbox_area(sample_detections):
    filtered_detections = sample_detections[sample_detections.area > 20000]
    assert len(filtered_detections) == 1


def test_filter_by_bbox_dimensions(sample_detections):
    filtered_detections = sample_detections[(sample_detections.bbox_width > 200) & (sample_detections.bbox_height > 200)]
    assert len(filtered_detections) == 1


def test_filter_by_mixed_conditions(sample_detections):
    filtered_detections = sample_detections[(sample_detections.confidence > 0.5) & (sample_detections.class_id == 0)]
    assert len(filtered_detections) == 1