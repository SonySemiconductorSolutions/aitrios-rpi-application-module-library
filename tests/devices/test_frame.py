#
# Copyright 2025 Sony Semiconductor Solutions Corp. All rights reserved.
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

import json

import numpy as np
import pytest

from modlib.devices.frame import Frame, IMAGE_TYPE
from modlib.models import COLOR_FORMAT, ROI, Detections


@pytest.fixture
def sample_frame() -> Frame:
    
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    
    detections = Detections(
        bbox=np.array([[1, 2, 3, 4]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([1], dtype=np.int32),
    )
    
    detections.tracker_id = np.array([5], dtype=np.int32)
    
    return Frame(
        timestamp="2025-01-01T00:00:00Z",
        image=image,
        image_type=IMAGE_TYPE.VGA,
        width=image.shape[1],
        height=image.shape[0],
        channels=image.shape[2],
        detections=detections,
        new_detection=True,
        fps=30.0,
        dps=10.0,
        color_format=COLOR_FORMAT.RGB,
        input_tensor=None,
        roi=ROI(left=0.1, top=0.2, width=0.3, height=0.4),
    )


def test_properties_and_setters(sample_frame: Frame):
    # image getter/setter
    assert sample_frame.image.shape == (480, 640, 3)
    new_image = np.zeros((480, 640, 3), dtype=np.uint8)
    sample_frame.image = new_image
    assert sample_frame.image is new_image

    # detections getter/setter
    assert isinstance(sample_frame.detections, Detections)
    new_detections = Detections()
    sample_frame.detections = new_detections
    assert sample_frame.detections is new_detections


def test_detections_unavailable_raises(sample_frame: Frame):
    sample_frame._detections = None
    with pytest.raises(ValueError):
        _ = sample_frame.detections


def test_json_roundtrip(sample_frame: Frame):
    json_dict = sample_frame.json()
    dumped = json.dumps(json_dict)
    assert isinstance(dumped, str)

    restored = Frame.from_json(json.loads(dumped))

    # basic fields
    assert restored.timestamp == sample_frame.timestamp
    assert restored.image_type == sample_frame.image_type
    assert restored.width == sample_frame.width
    assert restored.height == sample_frame.height
    assert restored.channels == sample_frame.channels
    assert restored.new_detection == sample_frame.new_detection
    assert restored.fps == sample_frame.fps
    assert restored.dps == sample_frame.dps
    assert restored.color_format == sample_frame.color_format

    # image
    assert restored.image.shape == sample_frame.image.shape
    assert (restored.image == sample_frame.image).all()

    # detections
    assert isinstance(restored.detections, Detections)
    assert restored.detections.bbox.tolist() == sample_frame.detections.bbox.tolist()
    assert restored.detections.confidence.tolist() == sample_frame.detections.confidence.tolist()
    assert restored.detections.class_id.tolist() == sample_frame.detections.class_id.tolist()
    assert restored.detections.tracker_id.tolist() == sample_frame.detections.tracker_id.tolist()

    # ROI
    assert restored.roi is not None
    assert restored.roi.json() == sample_frame.roi.json()
