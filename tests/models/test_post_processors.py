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

from modlib.models import Classifications, Detections, Poses, Segments
from modlib.models.post_processors import *


def test_pp_cls():
    output_tensor = [np.array([0.1, 0.4, 0.3, 0.2])]
    result = pp_cls(output_tensor)

    assert isinstance(result, Classifications)
    assert np.array_equal(result.confidence, np.array([0.4, 0.3, 0.2, 0.1]))
    assert np.array_equal(result.class_id, np.array([1, 2, 3, 0]))


def test_pp_cls_softmax():
    output_tensor = [np.log([0.1, 0.4, 0.3, 0.2]) - np.max([0.1, 0.4, 0.3, 0.2])]

    result = pp_cls_softmax(output_tensor)

    assert isinstance(result, Classifications)
    
    assert np.allclose(result.confidence, np.array([0.4, 0.3, 0.2, 0.1]), atol=1e-4)
    assert np.array_equal(result.class_id, np.array([1, 2, 3, 0]))


def test_pp_od_bcsn():
    
    output_tensors = [
        np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]), # Box coordinates
        np.array([1, 2]),       # Class IDs for 2 detections
        np.array([0.9, 0.8]),   # Scores for 2 detections
        np.array([2])           # Number of detections
    ]

    result = pp_od_bcsn(output_tensors)

    assert isinstance(result, Detections)
    assert np.array_equal(result.bbox, np.array([[0.1, 0.0, 0.3, 0.2], [0.5, 0.4, 0.7, 0.6]]))
    assert np.array_equal(result.class_id, np.array([1, 2]))
    assert np.array_equal(result.confidence, np.array([0.9, 0.8]))


def test_pp_od_bscn():
    
    output_tensors = [
        np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]), # Box coordinates
        np.array([0.9, 0.8]),   # Scores for 2 detections
        np.array([1, 2]),       # Class IDs for 2 detections
        np.array([2])           # Number of detections
    ]
   
    result = pp_od_bscn(output_tensors)

    assert isinstance(result, Detections)
    assert np.array_equal(result.bbox, np.array([[0.1, 0.0, 0.3, 0.2], [0.5, 0.4, 0.7, 0.6]]))
    assert np.array_equal(result.confidence, np.array([0.9, 0.8]))
    assert np.array_equal(result.class_id, np.array([1, 2]))


def test_pp_od_efficientdet_lite0():

    output_tensors = [
        np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]) * 320, # Box coordinates
        np.array([0.9, 0.8]),   # Scores for 2 detections
        np.array([1, 2]),       # Class IDs for 2 detections
        np.array([2])           # Number of detections
    ]
   
    result = pp_od_efficientdet_lite0(output_tensors)

    assert isinstance(result, Detections)
    assert np.array_equal(result.bbox, np.array([[0.1, 0.0, 0.3, 0.2], [0.5, 0.4, 0.7, 0.6]]))
    assert np.array_equal(result.confidence, np.array([0.9, 0.8]))
    assert np.array_equal(result.class_id, np.array([1, 2]))


def test_pp_posenet():
    # TODO
    pass


def test_pp_segment():
    # Create a mock output tensor: 320x320 image with random int8 values between 0 and 20
    mock_output = [np.random.randint(0, 21, size=(320, 320), dtype=np.int8)]

    result = pp_segment(mock_output)

    assert isinstance(result, Segments)
    assert result.mask.shape == (320, 320)
    assert result.mask.dtype == np.int8

    assert all(1 <= idx <= 20 for idx in result.indeces)
    assert len(result.indeces) == len(set(result.indeces))  # Ensure uniqueness
    assert len(result.indeces) > 0 and len(result.indeces) <= 21 - 0
    assert 0 not in result.indeces