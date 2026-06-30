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

import numpy as np
import pytest

from modlib.models.results import InstanceSegments, ROI


def _roi_expand_dense_stack(dense: np.ndarray, roi: ROI) -> np.ndarray:
    """Reference implementation of the former dense ROI compensation (N, h, w)."""
    if roi == (0, 0, 1, 1):
        return dense
    n, h, w = dense.shape
    out_h, out_w = int(h / roi[3]), int(w / roi[2])
    h_start, w_start = int(roi[1] * h / roi[3]), int(roi[0] * w / roi[2])
    start_h, start_w = max(0, -h_start), max(0, -w_start)
    out_start_h, out_start_w = max(0, h_start), max(0, w_start)
    delta_h = min(h - start_h, out_h - out_start_h)
    delta_w = min(w - start_w, out_w - out_start_w)
    new_masks = np.zeros((n, out_h, out_w), dtype=dense.dtype)
    new_masks[:, out_start_h : out_start_h + delta_h, out_start_w : out_start_w + delta_w] = dense[
        :, start_h : start_h + delta_h, start_w : start_w + delta_w
    ]
    return new_masks


def test_dense_init_compress_decompress_roundtrip():
    dense = np.zeros((2, 12, 16), dtype=np.uint8)
    dense[0, 2:8, 3:10] = 1
    dense[1, 6:11, 8:14] = 1
    bbox = InstanceSegments._bbox_from_dense_masks(dense)
    inst = InstanceSegments(mask=dense, bbox=bbox, class_id=np.array([3, 7], dtype=np.int32), confidence=np.array([0.7, 0.9]))
    assert len(inst._mask_crops) == 2
    assert inst.mask_shape == (12, 16)
    restored = inst.mask
    assert restored.shape == dense.shape
    assert np.array_equal(restored, dense)


def test_compressed_constructor_requires_mask_shape():
    crop = np.ones((2, 3), dtype=np.uint8)
    bbox = np.array([[0.1, 0.1, 0.5, 0.5]], dtype=np.float32)
    with pytest.raises(ValueError, match="mask_shape"):
        InstanceSegments(mask=[crop], bbox=bbox, class_id=np.array([1]), confidence=np.array([1.0]))


def test_compressed_constructor_matches_dense():
    dense = np.zeros((1, 10, 12), dtype=np.uint8)
    dense[0, 4:8, 5:11] = 1
    bbox = np.array([[0.4, 0.35, 0.95, 0.85]], dtype=np.float32)
    ref = InstanceSegments(mask=dense, bbox=bbox, class_id=np.array([2]), confidence=np.array([0.5]))
    inst = InstanceSegments(
        mask=list(ref._mask_crops),
        bbox=bbox,
        mask_shape=ref.mask_shape,
        class_id=np.array([2]),
        confidence=np.array([0.5]),
    )
    assert np.array_equal(inst.mask, ref.mask)


def test_getitem_slice():
    dense = np.zeros((3, 8, 8), dtype=np.uint8)
    for i in range(3):
        dense[i, i + 1 : i + 3, i + 1 : i + 3] = 1
    bbox = InstanceSegments._bbox_from_dense_masks(dense)
    inst = InstanceSegments(
        mask=dense,
        bbox=bbox,
        class_id=np.array([10, 20, 30]),
        confidence=np.array([0.1, 0.2, 0.3]),
    )
    sub = inst[::2]
    assert len(sub) == 2
    assert np.array_equal(sub.class_id, np.array([10, 30]))


def test_compensate_for_roi_matches_dense_reference():
    dense = np.zeros((1, 8, 8), dtype=np.uint8)
    dense[0, 2:6, 2:6] = 1
    bbox = InstanceSegments._bbox_from_dense_masks(dense)
    roi = ROI(left=0.0, top=0.0, width=0.5, height=0.5)
    inst = InstanceSegments(mask=dense.copy(), bbox=bbox, class_id=np.array([1]), confidence=np.array([1.0]))
    inst.compensate_for_roi(roi)
    expected = _roi_expand_dense_stack(dense, roi)
    assert np.array_equal(inst.mask, expected)
    assert inst.mask_shape == expected.shape[1:]
    for i, crop in enumerate(inst.mask_crops):
        yi1, yi2, xi1, xi2 = InstanceSegments._bbox_to_slices(inst.bbox[i], inst.mask_shape[0], inst.mask_shape[1])
        assert crop.shape == (yi2 - yi1, xi2 - xi1)


def test_compensate_for_roi_multi_instance_clipping_shapes_stay_consistent():
    dense = np.zeros((3, 10, 10), dtype=np.uint8)
    dense[0, 1:4, 1:4] = 1
    dense[1, 5:9, 6:10] = 1
    dense[2, 0:2, 8:10] = 1
    bbox = InstanceSegments._bbox_from_dense_masks(dense)
    inst = InstanceSegments(mask=dense.copy(), bbox=bbox, class_id=np.array([1, 2, 3]), confidence=np.array([0.5, 0.7, 0.9]))
    roi = ROI(left=0.25, top=0.0, width=0.75, height=0.75)

    inst.compensate_for_roi(roi)

    assert len(inst.mask_crops) == 3
    assert inst.bbox.shape == (3, 4)
    restored = inst.mask
    assert restored.shape[0] == 3
    for i, crop in enumerate(inst.mask_crops):
        yi1, yi2, xi1, xi2 = InstanceSegments._bbox_to_slices(inst.bbox[i], inst.mask_shape[0], inst.mask_shape[1])
        assert crop.shape == (yi2 - yi1, xi2 - xi1)


def test_to_segments():
    dense = np.zeros((2, 6, 6), dtype=np.uint8)
    dense[0, 1:4, 1:4] = 1
    dense[1, 2:5, 2:5] = 1
    bbox = InstanceSegments._bbox_from_dense_masks(dense)
    inst = InstanceSegments(
        mask=dense,
        bbox=bbox,
        class_id=np.array([5, 9], dtype=np.uint8),
        confidence=np.array([0.3, 0.9]),
    )
    seg = inst.to_segments()
    overlap = (dense[0] > 0) & (dense[1] > 0)
    assert seg.mask.shape == (6, 6)
    assert np.all(seg.mask[overlap] == 9)


def test_oriented_bbox_runs():
    cv2 = pytest.importorskip("cv2")
    dense = np.zeros((1, 32, 32), dtype=np.uint8)
    cv2.rectangle(dense[0], (8, 8), (23, 19), 1, thickness=-1)
    bbox = InstanceSegments._bbox_from_dense_masks(dense)
    inst = InstanceSegments(mask=dense, bbox=bbox, class_id=np.array([1]), confidence=np.array([1.0]))
    obb = inst.oriented_bbox()
    assert len(obb) >= 1
