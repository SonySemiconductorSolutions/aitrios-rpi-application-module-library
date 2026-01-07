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

from typing import List, Tuple

import numpy as np
from modlib.models.post_processors import cpp_post_processors

from ..results import Anomaly, Classifications, Detections, Poses, Segments, InstanceSegments
from .higherhrnet import postprocess_higherhrnet
from .yolo import postprocess_yolov8_detection, postprocess_yolov8_keypoints


def pp_cls(output_tensors: List[np.ndarray]) -> Classifications:
    """
    Performs post-processing on a Classification result tensor.

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed classification detections.
    """
    t = np.squeeze(output_tensors[0])

    # Creating sorted scores and indices arrays
    # Sorting the indices based on scores, negative for descending order
    sorted_indices = np.argsort(-t)

    return Classifications(confidence=t[sorted_indices], class_id=sorted_indices)


def pp_cls_softmax(output_tensors: List[np.ndarray]) -> Classifications:
    """
    Performs post-processing on a Classification result tensor that requires an additional softmax.

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed classification detections with softmax-applied confidence scores.
    """
    t = np.squeeze(output_tensors[0])

    # Softmax
    y = np.exp(t - np.expand_dims(np.max(t, axis=-1), axis=-1))
    np_output = y / np.expand_dims(np.sum(y, axis=-1), axis=-1)

    # Creating sorted scores and indices arrays
    # Sorting the indices based on scores, negative for descending order
    sorted_indices = np.argsort(-np_output)

    return Classifications(confidence=np_output[sorted_indices], class_id=sorted_indices)


def pp_od_bcsn(output_tensors: List[np.ndarray]) -> Detections:
    """
    Performs post-processing on an Object Detection result tensor.
    Output tensor order: Boxes - Classes - Scores - Number

    Args:
        output_tensors: Resulting output tensor to be processed.

    Returns:
        The post-processed object detection detections.
    """

    n_detections = int(output_tensors[3][0])

    return Detections(
        bbox=output_tensors[0][:n_detections][:, [1, 0, 3, 2]],
        class_id=np.array(output_tensors[1][:n_detections], dtype=np.uint16),
        confidence=output_tensors[2][:n_detections],
    )


def pp_od_bscn(output_tensors: List[np.ndarray]) -> Detections:
    """
    Performs post-processing on an Object Detection result tensor.
    Output tensor order: Boxes - Scores - Classes - Number

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed object detection detections.
    """

    n_detections = int(output_tensors[3][0])

    return Detections(
        bbox=output_tensors[0][:n_detections][:, [1, 0, 3, 2]],
        class_id=np.array(output_tensors[2][:n_detections], dtype=np.uint16),
        confidence=output_tensors[1][:n_detections],
    )


def pp_od_yolo_ultralytics(output_tensors: List[np.ndarray], input_tensor_sz: int = 640) -> Detections:
    """
    Performs post-processing on an Object Detection result tensor.
    In this case the model comes from Ultralytics YOLOv8n/YOLO11n model exported
    for imx using ultralytics tools (onnx model).
    Compared with `pp_od_bscn`:
    - bbox xy order is different
    - bboxes are scaled to input tensor size

    Output tensor order: Boxes - Scores - Classes - Number

    Args:
        output_tensors: Resulting output tensors to be processed.
        input_tensor_sz: Input tensor size, default 640.

    Returns:
        The post-processed object detection detections.
    """
    n_detections = int(output_tensors[3])
    detections = Detections(
        bbox=output_tensors[0][:n_detections],
        class_id=np.array(output_tensors[2][:n_detections], dtype=np.uint16),
        confidence=output_tensors[1][:n_detections],
    )
    detections.bbox /= input_tensor_sz
    return detections


def pp_od_yolov8n(output_tensors: List[np.ndarray]) -> Detections:
    """
    Postprocess the outputs of a YOLOv8 model for object detection, without any internal post-processing in the model.
    Default post processing settings:
    - **conf:** Confidence threshold for bbox predictions. Default is a low 0.01 to allow further filtering.
    - **iou_thres:** IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS). Default is 0.7.
    - **max_out_dets:** Maximum number of output detections to keep after NMS. Default is 50.

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed object detection detections.
    """

    boxes, scores, classes = postprocess_yolov8_detection(
        outputs=[np.expand_dims(t, axis=0) for t in output_tensors],  # batch=1
        conf=0.01,
        iou_thres=0.7,
        max_out_dets=50,
    )[0]

    return Detections(
        bbox=np.array(boxes / 640)[:, [1, 0, 3, 2]],
        class_id=classes,
        confidence=scores,
    )


def pp_od_efficientdet_lite0(output_tensors: List[np.ndarray]) -> Detections:
    """
    Performs post-processing on an Object Detection result tensor specifically for EfficientDet-Lite0.

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed object detection detections, with bounding box coordinates normalized to a 320x320 scale.
    """

    detections = pp_od_bscn(output_tensors)
    detections.bbox /= 320
    return detections


def pp_posenet(output_tensors: List[np.ndarray]) -> Poses:
    """
    Performs post-processing on a Posenet result tensor.
    The output tensor is post processed by the posenet decoder handled with a binding to C++.
    The interface of this function: PosenetOutputDataType is populated with the decoded pose data, including:
        - Number of detections (n_detections).
        - Pose scores (pose_scores).
        - Keypoints for each detected pose (pose_keypoints).
        - Scores for each keypoint in the detected poses (pose_keypoint_scores).

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed pose estimation results.
    """
    n_detections, pose_scores, pose_keypoints, pose_keypoint_scores = cpp_post_processors.decode_poses_cpp(
        np.ascontiguousarray(output_tensors[0]),
        np.ascontiguousarray(output_tensors[1]),
        np.ascontiguousarray(output_tensors[2]),
    )

    # Normalize keypoints between 0 and 1
    input_tensor_size = (481, 353)
    keypoints = np.array(pose_keypoints)
    if n_detections > 0:
        keypoints[:, :, 0] /= input_tensor_size[0]
        keypoints[:, :, 1] /= input_tensor_size[1]

    return Poses(
        n_detections=n_detections,
        confidence=np.array(pose_scores),
        keypoints=keypoints,
        keypoint_scores=np.array(pose_keypoint_scores),
    )


def pp_personlab_py(
    output_tensors: List[np.ndarray],
    num_keypoints: int,
    edges: List[Tuple[int, int]],
    peak_threshold: float,
    nms_threshold: float,
    kp_radius: float,
) -> Poses:
    import cv2
    from scipy.ndimage import gaussian_filter, maximum_filter

    IN_HEIGHT = 353
    IN_WIDTH = 481

    def resize_tensors(t: np.ndarray, new_shape: list) -> np.ndarray:
        num_channels = t.shape[-1]
        resized_tensor = np.empty((new_shape[0], new_shape[1], num_channels), dtype=t.dtype)
        for c in range(num_channels):
            resized_tensor[:, :, c] = cv2.resize(
                t[:, :, c], (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR
            )

        return resized_tensor

    kp_maps, short_offsets, mid_offsets = [resize_tensors(t, (353, 481)) for i, t in enumerate(output_tensors)]

    def split_and_refine_mid_offsets(edges, mid_offsets, short_offsets):
        rev_edges = [edge[::-1] for edge in edges]
        edges_loop = edges + rev_edges

        output_mid_offsets = []
        for mid_idx, edge in enumerate(edges_loop):
            to_keypoint = edge[1]
            kp_short_offsets = short_offsets[:, :, 2 * to_keypoint : 2 * to_keypoint + 2]
            kp_mid_offsets = mid_offsets[:, :, 2 * mid_idx : 2 * mid_idx + 2]

            # Refine: perform a few steps of iterative refinement
            num_steps = 2
            for _ in range(num_steps):
                kp_mid_offsets += bilinear_sampler_numpy(kp_short_offsets, kp_mid_offsets)

            output_mid_offsets.append(kp_mid_offsets)

        r = np.concatenate(output_mid_offsets, axis=-1)
        return r

    def bilinear_sampler_numpy(x, v):
        H, W, C = x.shape
        # Pad the input image along height and width.
        pad_x = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="constant")

        # Split the velocity field into vx and vy.
        vx = v[..., 0]  # first of 2 channels
        vy = v[..., 1]  # second of 2 channels

        # Create a grid of indices corresponding to the original image (shifted for the padded image)
        h_arr, w_arr = np.meshgrid(np.arange(1, H + 1), np.arange(1, W + 1), indexing="ij")

        # Compute floor and ceil for sampling coordinates.
        vx0 = np.floor(vx)
        vy0 = np.floor(vy)
        vx1 = vx0 + 1
        vy1 = vy0 + 1

        # Compute absolute indices in the padded image.
        iy0 = vy0 + h_arr
        iy1 = vy1 + h_arr
        ix0 = vx0 + w_arr
        ix1 = vx1 + w_arr

        # Bound check: if any coordinate is out of bounds, set the index to 0.
        # (Valid padded indices are in the range [1, H] and [1, W])
        mask = (ix0 < 1) | (iy0 < 1) | (ix1 > W) | (iy1 > H)
        iy0 = np.where(mask, 0, iy0).astype(np.int32)
        iy1 = np.where(mask, 0, iy1).astype(np.int32)
        ix0 = np.where(mask, 0, ix0).astype(np.int32)
        ix1 = np.where(mask, 0, ix1).astype(np.int32)

        # Use advanced indexing to gather the four neighboring pixels.
        x00 = pad_x[iy0, ix0]  # Top-left
        x01 = pad_x[iy1, ix0]  # Bottom-left
        x10 = pad_x[iy0, ix1]  # Top-right
        x11 = pad_x[iy1, ix1]  # Bottom-right

        # Compute interpolation weights.
        dx = (vx - vx0)[..., np.newaxis]
        dy = (vy - vy0)[..., np.newaxis]
        w00 = (1.0 - dx) * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w10 = dx * (1.0 - dy)
        w11 = dx * dy

        # Combine the contributions.
        output = w00 * x00 + w01 * x01 + w10 * x10 + w11 * x11
        return output

    mid_offsets = split_and_refine_mid_offsets(edges, mid_offsets, short_offsets)

    # compute heatmaps
    def compute_heatmaps(kp_maps, short_offsets, kp_radius):
        heatmaps = []
        map_shape = kp_maps.shape[:2]
        idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1, 0, 2))
        for i in range(num_keypoints):
            this_kp_map = kp_maps[:, :, i : i + 1]
            votes = idx + short_offsets[:, :, 2 * i : 2 * i + 2]
            votes = np.reshape(np.concatenate([votes, this_kp_map], axis=-1), (-1, 3))
            r = accumulate_votes(votes, shape=map_shape) / (np.pi * kp_radius**2)
            heatmaps.append(r)

        return np.stack(heatmaps, axis=-1)

    def accumulate_votes(votes, shape):
        xs = votes[:, 0]
        ys = votes[:, 1]
        ps = votes[:, 2]
        tl = [np.floor(ys).astype("int32"), np.floor(xs).astype("int32")]
        tr = [np.floor(ys).astype("int32"), np.ceil(xs).astype("int32")]
        bl = [np.ceil(ys).astype("int32"), np.floor(xs).astype("int32")]
        br = [np.ceil(ys).astype("int32"), np.ceil(xs).astype("int32")]
        dx = xs - tl[1]
        dy = ys - tl[0]
        tl_vals = ps * (1.0 - dx) * (1.0 - dy)
        tr_vals = ps * dx * (1.0 - dy)
        bl_vals = ps * dy * (1.0 - dx)
        br_vals = ps * dy * dx
        data = np.concatenate([tl_vals, tr_vals, bl_vals, br_vals])
        I = np.concatenate([tl[0], tr[0], bl[0], br[0]])  # noqa
        J = np.concatenate([tl[1], tr[1], bl[1], br[1]])
        good_inds = np.logical_and(I >= 0, I < shape[0])
        good_inds = np.logical_and(good_inds, np.logical_and(J >= 0, J < shape[1]))

        heatmap = np.zeros(shape, dtype=np.float32)
        np.add.at(heatmap, (I[good_inds], J[good_inds]), data[good_inds])

        return heatmap

    H = compute_heatmaps(kp_maps=kp_maps, short_offsets=short_offsets, kp_radius=kp_radius)

    for i in range(num_keypoints):
        H[:, :, i] = gaussian_filter(H[:, :, i], sigma=2)

    # get keypoints
    def get_keypoints(heatmaps):
        keypoints = []
        for i in range(num_keypoints):
            peaks = maximum_filter(heatmaps[:, :, i], footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]]) == heatmaps[:, :, i]
            peaks = zip(*np.nonzero(peaks))
            keypoints.extend(
                [
                    {
                        "id": i,
                        "xy": np.array(peak[::-1]),
                        "conf": heatmaps[peak[0], peak[1], i],
                    }
                    for peak in peaks
                ]
            )
            keypoints = [kp for kp in keypoints if kp["conf"] > peak_threshold]

        return keypoints

    pred_kp = get_keypoints(H)

    # Group skeletons
    def group_skeletons(keypoints, mid_offsets):
        keypoints.sort(key=(lambda kp: kp["conf"]), reverse=True)
        skeletons = []
        dir_edges = edges + [edge[::-1] for edge in edges]

        skeleton_graph = {i: [] for i in range(num_keypoints)}
        for i in range(num_keypoints):
            for j in range(num_keypoints):
                if (i, j) in edges or (j, i) in edges:
                    skeleton_graph[i].append(j)
                    skeleton_graph[j].append(i)

        while len(keypoints) > 0:
            kp = keypoints.pop(0)
            if any([np.linalg.norm(kp["xy"] - s[kp["id"], :2]) <= 10 for s in skeletons]):
                continue
            this_skel = np.zeros((num_keypoints, 3))
            this_skel[kp["id"], :2] = kp["xy"]
            this_skel[kp["id"], 2] = kp["conf"]
            path = iterative_bfs(skeleton_graph, kp["id"])[1:]
            for edge in path:
                if this_skel[edge[0], 2] == 0:
                    continue
                mid_idx = dir_edges.index(edge)
                offsets = mid_offsets[:, :, 2 * mid_idx : 2 * mid_idx + 2]
                from_kp = tuple(np.round(this_skel[edge[0], :2]).astype("int32"))
                proposal = this_skel[edge[0], :2] + offsets[from_kp[1], from_kp[0], :]
                matches = [(i, keypoints[i]) for i in range(len(keypoints)) if keypoints[i]["id"] == edge[1]]
                matches = [match for match in matches if np.linalg.norm(proposal - match[1]["xy"]) <= nms_threshold]
                if len(matches) == 0:
                    continue
                matches.sort(key=lambda m: np.linalg.norm(m[1]["xy"] - proposal))
                to_kp = np.round(matches[0][1]["xy"]).astype("int32")
                to_kp_conf = matches[0][1]["conf"]
                keypoints.pop(matches[0][0])
                this_skel[edge[1], :2] = to_kp
                this_skel[edge[1], 2] = to_kp_conf

            skeletons.append(this_skel)

        return skeletons

    def iterative_bfs(graph, start, path=[]):
        """iterative breadth first search from start"""
        q = [(None, start)]
        visited = []
        while q:
            v = q.pop(0)
            if v[1] not in visited:
                visited.append(v[1])
                path = path + [v]
                q = q + [(v[1], w) for w in graph[v[1]]]
        return path

    skeletons = group_skeletons(keypoints=pred_kp, mid_offsets=mid_offsets)

    n_detections = len(skeletons)
    if n_detections != 0:
        keypoints = np.array(skeletons)[:, :, :2]
        keypoints[:, :, 1] /= IN_HEIGHT
        keypoints[:, :, 0] /= IN_WIDTH

        keypoint_scores = np.array(skeletons)[:, :, 2]  # NOTE normalize ? very low values
    else:
        keypoints, keypoint_scores = np.array([]), np.array([])

    return Poses(
        n_detections=n_detections,
        confidence=np.ones(n_detections),
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        bbox=None,
    )


def pp_personlab(
    output_tensors: List[np.ndarray],
    num_keypoints: int,
    edges: List[Tuple[int, int]],
    peak_threshold: float,
    nms_threshold: float,
    kp_radius: float,
) -> Poses:
    """
    Performs post-processing on a PersonLab result tensor.

    Args:
        output_tensors: Resulting output tensors to be processed.
        num_keypoints: Number of keypoints to detect in each object.
        edges: List of tuples defining the connections between keypoints.
        peak_threshold: Confidence threshold for keypoint detection.
        nms_threshold: Non-maximum suppression threshold for keypoint grouping.
        kp_radius: Radius of the discs around the keypoints. Used for computing the ground truth
            and computing the losses. (Recommended to be a multiple of the output stride.)

    Returns:
        The post-processed pose estimation results, containing detected keypoints and their scores.
    """

    IN_HEIGHT = 353
    IN_WIDTH = 481

    skeletons = cpp_post_processors.decode_personlab_cpp(
        np.ascontiguousarray(output_tensors[0]),
        np.ascontiguousarray(output_tensors[1]),
        np.ascontiguousarray(output_tensors[2]),
        num_keypoints,
        edges,
        peak_threshold,
        nms_threshold,
        kp_radius,
    )

    n_detections = len(skeletons)
    if n_detections != 0:
        keypoints = np.array(skeletons)[:, :, :2]
        keypoints[:, :, 1] /= IN_HEIGHT
        keypoints[:, :, 0] /= IN_WIDTH

        keypoint_scores = np.array(skeletons)[:, :, 2]  # NOTE normalize ? very low values
    else:
        keypoints, keypoint_scores = np.array([]), np.array([])

    return Poses(
        n_detections=n_detections,
        confidence=np.ones(n_detections),
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        bbox=None,
    )


def pp_higherhrnet(output_tensors: List[np.ndarray]) -> Poses:
    """
    Performs post-processing on a HigherhrNet result tensor.

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed pose estimation results.
    """

    kpts, scores = postprocess_higherhrnet(
        outputs=[np.expand_dims(t, axis=0) for t in output_tensors],  # batch=1
        img_size=(480, 640),
        img_w_pad=(0, 0),
        img_h_pad=(0, 0),
        detection_threshold=0.3,
        network_postprocess=True,
    )

    keypoints = np.reshape(kpts, [kpts.shape[0], 17, 3])
    keypoints[:, :, 0] /= 640
    keypoints[:, :, 1] /= 480

    return Poses(
        n_detections=len(scores),
        confidence=scores,
        keypoints=keypoints[:, :, :2],
        keypoint_scores=keypoints[:, :, 2],
    )


def pp_yolov8n_pose(output_tensors: List[np.ndarray]) -> Poses:
    """
    Performs post-processing on a raw YOLOv8n-pose result tensor.

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed pose estimation results.
    """

    boxes, scores, kpts = postprocess_yolov8_keypoints(
        outputs=[np.expand_dims(t, axis=0) for t in output_tensors],  # batch=1
        conf=0.01,
        iou_thres=0.3,
        max_out_dets=10,
    )

    keypoints = np.reshape(kpts, [kpts.shape[0], 17, 3])
    keypoints[:, :, 0] /= 640
    keypoints[:, :, 1] /= 640

    n_detections = len(scores)
    return Poses(
        n_detections=n_detections,
        confidence=scores,
        keypoints=keypoints[:, :, :2],
        keypoint_scores=keypoints[:, :, 2],
        bbox=np.reshape(boxes / 640, [n_detections, 4])[:, [1, 0, 3, 2]],
    )


def pp_yolo_pose_ultralytics(
    output_tensors: List[np.ndarray], input_tensor_sz: int = 640, num_keypoints: int = 17
) -> Poses:
    """
    Performs post-processing on an Ultralytics YOLO-pose result tensor.
    In this case the model comes from the Ultralytics YOLOv8n-pose/YOLO11n-pose model exported
    for imx using ultralytics tools (onnx model).

    Args:
        output_tensors: Resulting output tensors to be processed.
        input_tensor_sz: Input tensor size, default 640.
        num_keypoints: Number of keypoints to detect in each object.

    Returns:
        The post-processed pose estimation results.
    """
    boxes, scores, class_ids, kpts = output_tensors

    keypoints = np.reshape(kpts, [kpts.shape[0], num_keypoints, 3])
    keypoints[:, :, 0] /= input_tensor_sz
    keypoints[:, :, 1] /= input_tensor_sz

    n_detections = len(scores)
    return Poses(
        n_detections=n_detections,
        confidence=scores,
        keypoints=keypoints[:, :, :2],
        keypoint_scores=keypoints[:, :, 2],
        bbox=boxes / input_tensor_sz,
    )


def pp_segment(output_tensors: List[np.ndarray]) -> Segments:
    """
    Performs post-processing on a Segmentation model result tensor.

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed segmentation results.
    """
    return Segments(mask=output_tensors[0])


def pp_yolo_segment_ultralytics(
    output_tensors: List[np.ndarray], input_tensor_sz: int = 640, max_detections: int = 10, threshold: float = 0.5
) -> Segments:
    """
    Performs post-processing on an Ultralytics YOLO-segments result tensor.
    In this case the model comes from the Ultralytics YOLOv8n-segments/YOLO11n-segments model exported
    for imx using ultralytics tools (onnx model).

    Args:
        output_tensors: Resulting output tensors to be processed.
        input_tensor_sz: Input tensor size, default 640.
        max_detections: Maximum number of detections, default is 10.
        threshold: Threshold value of filtering the mask, default is 0.5.

    Returns:
        The post-processed segmentation results.
    """

    def sigmoid(x):
        x = np.array(x)  # Ensure x is a NumPy array for element-wise operations
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    n = len(np.array(output_tensors[2]))

    boxes = output_tensors[0][:max_detections]
    classes = output_tensors[2][:max_detections].astype(np.uint16)
    scores = output_tensors[1][:max_detections]
    mask_coeffs = output_tensors[3][:max_detections]
    masks_proto = output_tensors[4][:n]

    c, mh, mw = masks_proto.shape
    masks_proto_flattered = masks_proto.reshape(c, -1)
    mask = np.dot(mask_coeffs, masks_proto_flattered)
    mask = sigmoid(mask)
    mask = mask.reshape(-1, mh, mw)
    binary_mask = (mask > threshold).astype(np.uint8)

    # crop
    scale_x = mw / input_tensor_sz
    scale_y = mh / input_tensor_sz
    boxes_scaled = boxes.copy()
    boxes_scaled[:, 0] *= scale_x
    boxes_scaled[:, 1] *= scale_y
    boxes_scaled[:, 2] *= scale_x
    boxes_scaled[:, 3] *= scale_y

    boxes_scaled = boxes_scaled.astype(int)

    x1, y1, x2, y2 = np.split(boxes_scaled[:, :, None], 4, axis=1)

    r = np.arange(mw, dtype=x1.dtype)[None, None, :]
    c = np.arange(mh, dtype=x1.dtype)[None, :, None]

    cropped_mask = binary_mask * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))

    boxes /= input_tensor_sz

    return InstanceSegments(mask=cropped_mask, bbox=boxes, class_id=classes, confidence=scores)


def pp_anomaly(output_tensors: List[np.ndarray]) -> Anomaly:
    """
    Performs post-processing on an Anomaly detection result tensor.

    Args:
        output_tensors: Resulting output tensors to be processed.

    Returns:
        The post-processed anomaly results, containing an anomaly score and an anomaly heatmap.
    """
    np_output = np.squeeze(output_tensors)
    return Anomaly(score=np_output[0, 0, 1], heatmap=np_output[:, :, 0])
