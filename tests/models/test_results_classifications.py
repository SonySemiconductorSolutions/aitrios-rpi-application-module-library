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

from modlib.models import Classifications


@pytest.fixture
def sample_classifications():
    c = Classifications()
    c.confidence = np.array([0.6, 0.8, 0.4])
    c.class_id = np.array([0, 1, 0])
    return c


def test_init(sample_classifications):
    assert len(sample_classifications.confidence) == 3
    assert len(sample_classifications.class_id) == 3


def test_len(sample_classifications):
    assert len(sample_classifications) == 3


def test_iter(sample_classifications):
    for i, (confidence, class_id) in enumerate(sample_classifications):
        assert confidence == sample_classifications.confidence[i]
        assert class_id == sample_classifications.class_id[i]


def test_add(sample_classifications):
    c1 = Classifications()
    c1.confidence = np.array([0.9])
    c1.class_id = np.array([1])

    c = sample_classifications + c1
    assert np.array_equal(c.confidence, np.concatenate([sample_classifications.confidence, c1.confidence]))
    assert np.array_equal(c.class_id, np.concatenate([sample_classifications.class_id, c1.class_id]))


def test_add_type_error():
    c = Classifications()
    with pytest.raises(TypeError):
        c = c + 1


def test_getitem_int(sample_classifications):
    c = sample_classifications[1]
    assert len(c) == 1
    assert c.confidence[0] == sample_classifications.confidence[1]
    assert c.class_id[0] == sample_classifications.class_id[1]


def test_getitem_slice(sample_classifications):
    c = sample_classifications[0:2]
    assert len(c) == 2
    assert np.array_equal(c.confidence, sample_classifications.confidence[0:2])
    assert np.array_equal(c.class_id, sample_classifications.class_id[0:2])


def test_getitem_list(sample_classifications):
    indices = [0, 2]
    c = sample_classifications[indices]
    assert len(c) == 2
    assert np.array_equal(c.confidence, sample_classifications.confidence[indices])
    assert np.array_equal(c.class_id, sample_classifications.class_id[indices])


# Representations
def test_str(sample_classifications):
    expected_str = f"Classifications(class_id:\t {sample_classifications.class_id}, \tconfidence:\t {sample_classifications.confidence})"
    assert str(sample_classifications) == expected_str


def test_json(sample_classifications):
    json_dict = sample_classifications.json()
    assert json_dict["confidence"] == sample_classifications.confidence.tolist()
    assert json_dict["class_id"] == sample_classifications.class_id.tolist()


# Filtering
def test_filter_by_id(sample_classifications):
    filtered_classifications = sample_classifications[sample_classifications.class_id == 0]
    assert len(filtered_classifications) == 2
    assert np.all(filtered_classifications.class_id == 0)


def test_filter_by_confidence(sample_classifications):
    filtered_classifications = sample_classifications[sample_classifications.confidence > 0.5]
    assert len(filtered_classifications) == 2


def test_filter_by_mixed_conditions(sample_classifications):
    filtered_classifications = sample_classifications[(sample_classifications.confidence > 0.5) & (sample_classifications.class_id == 0)]
    assert len(filtered_classifications) == 1