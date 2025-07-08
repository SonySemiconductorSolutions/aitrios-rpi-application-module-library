# https://github.com/ifzhang/ByteTrack/blob/main/yolox/tracker/matching.py

"""
MIT License

Copyright (c) 2021 Yifu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import lap
import numpy as np


def bbox_overlaps(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        atlbrs: An array of shape (M, 4) representing the first set of bounding boxes.
        btlbrs: An array of shape (N, 4) representing the second set of bounding boxes.

    Returns:
        The `ious`, a matrix of shape (M, N) where element (i, j) is the IoU between the ith bounding box in atlbrs
        and the jth bounding box in btlbrs.
    """

    # Expand dimensions to allow broadcasting
    a_areas = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    b_areas = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])

    # Broadcast the bounding boxes' coordinates for easier computation
    inter_x1 = np.maximum(atlbrs[:, None, 0], btlbrs[:, 0])
    inter_y1 = np.maximum(atlbrs[:, None, 1], btlbrs[:, 1])
    inter_x2 = np.minimum(atlbrs[:, None, 2], btlbrs[:, 2])
    inter_y2 = np.minimum(atlbrs[:, None, 3], btlbrs[:, 3])

    # Compute the area of intersection rectangle
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Compute the area of union
    union_area = (a_areas[:, None] + b_areas) - inter_area

    # Compute the IoU
    ious = inter_area / union_area

    return ious


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []

    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)

    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """
    Compute cost based on IoU

    Args:
        atlbrs (List[STrack.tlbr] | np.ndarray): atlbrs
        btlbrs (List[STrack.tlbr] | np.ndarray): btlbrs

    Returns:
        ious
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_overlaps(np.ascontiguousarray(atlbrs, dtype=np.float64), np.ascontiguousarray(btlbrs, dtype=np.float64))

    return ious


def iou_distance(atracks, btracks) -> np.ndarray:
    """
    Compute cost based on IoU

    Args:
        atracks (List[STrack]): atracks
        btracks (List[STrack]): atracks

    Returns:
        The cost_matrix
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
