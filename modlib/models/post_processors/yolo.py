#
# BSD 2-Clause License
#
# Copyright (c) 2021, Raspberry Pi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
This code is based on:
https://github.com/ultralytics/ultralytics
"""

from enum import Enum
from typing import List, Tuple

import numpy as np


def nms(dets: np.ndarray, scores: np.ndarray, iou_thres: float = 0.55, max_out_dets: int = 50) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on detected bounding boxes.

    Args:
        dets: Array of bounding box coordinates of shape (N, 4) representing [y1, x1, y2, x2].
        scores: Array of confidence scores associated with each bounding box.
        iou_thres: IoU threshold for NMS. Default is 0.5.
        max_out_dets: Maximum number of output detections to keep. Default is 300.

    Returns:
        List of indices representing the indices of the bounding boxes to keep after NMS.
    """
    y1, x1 = dets[:, 0], dets[:, 1]
    y2, x2 = dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep[:max_out_dets]


def combined_nms(batch_boxes, batch_scores, iou_thres: float = 0.65, conf: float = 0.55, max_out_dets: int = 50):
    nms_results = []
    for boxes, scores in zip(batch_boxes, batch_scores):
        xc = np.argmax(scores, 1)
        xs = np.amax(scores, 1)
        x = np.concatenate([boxes, np.expand_dims(xs, 1), np.expand_dims(xc, 1)], 1)

        xi = xs > conf
        x = x[xi]

        x = x[np.argsort(-x[:, 4])[:8400]]
        scores = x[:, 4]
        x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
        offset = x[:, 5] * 640
        boxes = x[..., :4] + np.expand_dims(offset, 1)

        # Original post-processing part
        valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
        x = x[valid_indexs]
        nms_classes = x[:, 5]
        nms_bbox = x[:, :4]
        nms_scores = x[:, 4]

        nms_results.append((nms_bbox, nms_scores, nms_classes))

    return nms_results


def combined_nms_seg(
    batch_boxes, batch_scores, batch_masks, iou_thres: float = 0.5, conf: float = 0.001, max_out_dets: int = 300
):
    nms_results = []
    for boxes, scores, masks in zip(batch_boxes, batch_scores, batch_masks):
        # Compute maximum scores and corresponding class indices
        class_indices = np.argmax(scores, axis=1)
        max_scores = np.amax(scores, axis=1)
        detections = np.concatenate(
            [boxes, np.expand_dims(max_scores, axis=1), np.expand_dims(class_indices, axis=1)], axis=1
        )

        # Swap the position of the two dimensions (32, 8400) to (8400, 32)
        masks = np.transpose(masks, (1, 0))
        # Filter out detections below the confidence threshold
        valid_detections = max_scores > conf

        if np.all(valid_detections is False):
            nms_results.append((np.ndarray(0), np.ndarray(0), np.ndarray(0), np.ndarray(0)))
        else:
            detections = detections[valid_detections]
            masks = masks[valid_detections]

            # Sort detections by score in descending order
            sorted_indices = np.argsort(-detections[:, 4])
            detections = detections[sorted_indices]
            masks = masks[sorted_indices]

            detections[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(detections[..., :4], BoxFormat.XC_YC_W_H)

            # Perform class-wise NMS
            unique_classes = np.unique(detections[:, 5])
            final_indices = []

            for cls in unique_classes:
                cls_indices = np.where(detections[:, 5] == cls)[0]
                cls_boxes = detections[cls_indices, :4]
                cls_scores = detections[cls_indices, 4]
                cls_valid_indices = nms(cls_boxes, cls_scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
                final_indices.extend(cls_indices[cls_valid_indices])

            final_indices = np.array(final_indices)
            final_detections = detections[final_indices]
            final_masks = masks[final_indices]

            # Extract class indices, bounding boxes, and scores
            nms_classes = final_detections[:, 5]
            nms_bbox = final_detections[:, :4]
            nms_scores = final_detections[:, 4]

            # Append results including masks
            nms_results.append((nms_bbox, nms_scores, nms_classes, final_masks))
    return nms_results


class BoxFormat(Enum):
    """
    Enumeration of different bounding box formats used in object detection.
    """

    YMIM_XMIN_YMAX_XMAX = "ymin_xmin_ymax_xmax"
    XMIM_YMIN_XMAX_YMAX = "xmin_ymin_xmax_ymax"
    XMIN_YMIN_W_H = "xmin_ymin_width_height"
    XC_YC_W_H = "xc_yc_width_height"


def convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format: BoxFormat):
    """
    Changes the box from one format to another when needed (XMIN_YMIN_W_H --> YMIM_XMIN_YMAX_XMAX)
    """
    if len(boxes) == 0:
        return boxes
    elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
        return boxes
    elif orig_format == BoxFormat.XMIN_YMIN_W_H:
        boxes[:, 2] += boxes[:, 0]  # convert width to xmax
        boxes[:, 3] += boxes[:, 1]  # convert height to ymax
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XC_YC_W_H:
        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
        new_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
        new_boxes[:, 2] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
        new_boxes[:, 3] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
        return new_boxes
    else:
        raise Exception("Unsupported boxes format")


def postprocess_yolov8_detection(
    outputs: Tuple[np.ndarray, np.ndarray], conf: float = 0.3, iou_thres: float = 0.7, max_out_dets: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess the outputs of a YOLOv8 model for object detection

    Args:
        outputs: Tuple containing the model outputs for bounding boxes and class predictions.
        conf: Confidence threshold for bounding box predictions. Default is 0.3
        iou_thres: IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS). Default is 0.7.
        max_out_dets: Maximum number of output detections to keep after NMS. Default is 50.

    Returns:
        Tuple containing the post-processed bounding boxes, their corresponding scores, and categories.
    """
    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls = outputs
    dbox = dist2bbox_yolo_v8(y_bb, a, xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)

    xd = detect_out.transpose([0, 2, 1])

    return combined_nms(xd[..., :4], xd[..., 4:84], iou_thres, conf, max_out_dets)


def postprocess_yolov8_keypoints(
    outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    conf: float = 0.3,
    iou_thres: float = 0.7,
    max_out_dets: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess the outputs of a YOLOv8 model for object detection and pose estimation.

    Args:
        outputs: Tuple containing the model outputs for bounding boxes, class predictions, and keypoint predictions.
        conf: Confidence threshold for bounding box predictions. Default is 0.3
        iou_thres: IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS). Default is 0.7.
        max_out_dets: Maximum number of output detections to keep after NMS. Default is 300.

    Returns:
        Tuple containing the post-processed bounding boxes, their corresponding scores, and keypoints.
    """
    kpt_shape = (17, 3)
    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls, kpts = outputs
    dbox = dist2bbox_yolo_v8(y_bb, a, xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)
    # additional part for pose estimation
    ndim = kpt_shape[1]
    pred_kpt = kpts.copy()
    if ndim == 3:
        pred_kpt[:, 2::3] = 1 / (1 + np.exp(-pred_kpt[:, 2::3]))  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
    pred_kpt[:, 0::ndim] = (pred_kpt[:, 0::ndim] * 2.0 + (a[0] - 0.5)) * s
    pred_kpt[:, 1::ndim] = (pred_kpt[:, 1::ndim] * 2.0 + (a[1] - 0.5)) * s

    x = np.concatenate([detect_out.transpose([2, 1, 0]).squeeze(), pred_kpt.transpose([2, 1, 0]).squeeze()], 1)
    x = x[(x[:, 4] > conf)]
    x = x[np.argsort(-x[:, 4])[:8400]]
    x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
    boxes = x[..., :4]
    scores = x[..., 4]

    # Original post-processing part
    valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
    x = x[valid_indexs]
    nms_bbox = x[:, :4]
    nms_scores = x[:, 4]
    nms_kpts = x[:, 5:]

    return nms_bbox, nms_scores, nms_kpts


def postprocess_yolov8_inst_seg(
    outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    conf: float = 0.001,
    iou_thres: float = 0.7,
    max_out_dets: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls, ymask_weights, y_masks = outputs
    dbox = dist2bbox_yolo_v8(y_bb, a, xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)

    xd = detect_out.transpose([0, 2, 1])
    nms_bbox, nms_scores, nms_classes, ymask_weights = combined_nms_seg(
        xd[..., :4], xd[..., 4:84], ymask_weights, iou_thres, conf, max_out_dets
    )[0]
    if len(nms_scores) == 0:
        final_masks = y_masks
    else:
        y_masks = y_masks.squeeze(0)
        ymask_weights = ymask_weights.transpose(1, 0)
        final_masks = np.tensordot(ymask_weights, y_masks, axes=([0], [0]))

    return nms_bbox, nms_scores, nms_classes, final_masks


def make_anchors_yolo_v8(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i], feats[i]
        sx = np.arange(stop=w) + grid_cell_offset  # shift x
        sy = np.arange(stop=h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def dist2bbox_yolo_v8(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox
