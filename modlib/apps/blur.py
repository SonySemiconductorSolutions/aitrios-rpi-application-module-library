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

from typing import Optional

import cv2
import numpy as np

from modlib.devices.frame import IMAGE_TYPE, Frame
from modlib.models import Detections, Poses


def blur_object(frame: Frame, detections: Detections, intensity: Optional[int] = 30) -> np.ndarray:
    """
    Blurs the objects detected on a given frame.

    Args:
        frame: The current frame to blur the object on.
        detections: The object detections.
        intensity: The intensity of the blur applied. Defaults to 30.

    Returns:
        The blurred frame.image
    """

    if not isinstance(detections, Detections) and not isinstance(detections, Poses):
        raise ValueError("Input `detections` should be of type Detections")

    # NOTE: Compensating for any introduced modified region of interest (ROI)
    # to ensure that the blurred region is displayed correctly on top of the current `frame.image`.
    if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
        detections.compensate_for_roi(frame.roi)

    h, w, _ = frame.image.shape
    if detections.bbox.any():
        for detection in detections.bbox:
            x1, y1, x2, y2 = int(detection[0] * w), int(detection[1] * h), int(detection[2] * w), int(detection[3] * h)
            region = frame.image[y1:y2, x1:x2]
            blurred_region = cv2.blur(region, (intensity, intensity))
            frame.image[y1:y2, x1:x2] = blurred_region

    return frame.image


def blur_face(frame: Frame, poses: Poses, intensity: Optional[int] = 30, padding: Optional[int] = 5) -> np.ndarray:
    """
    Blurs the face keypoints region detected on a given frame.

    Args:
        frame: The current frame to blur the object on.
        poses: The poses key points.
        intensity: The intensity of the blur applied. Defaults to 30.
        padding: The padding added to blur bounding box to cover the face. Defaults to 5

    Returns:
        The blurred frame.image
    """

    if not isinstance(poses, Poses):
        raise ValueError("Detections must be of type Poses.")

    # NOTE: Compensating for any introduced modified region of interest (ROI)
    # to ensure that the blurred region is displayed correctly on top of the current `frame.image`.
    if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
        poses.compensate_for_roi(frame.roi)

    h, w, _ = frame.image.shape

    def get_face_keypoints(poses, pose_idx, keypoint_idx, w, h):
        x = int(poses.keypoints[pose_idx, keypoint_idx, 0] * w)
        y = int(poses.keypoints[pose_idx, keypoint_idx, 1] * h)
        return x, y

    for i in range(poses.n_detections):
        x_min, y_min, x_max, y_max = w, h, 0, 0
        # get face keypoints
        for j in range(5):
            x, y = get_face_keypoints(poses, i, j, w, h)
            if (x is not None and y is not None) and (x != 0 and y != 0):
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

                # keep the values within the bounds of frame.image
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

        if x_min < x_max and y_min < y_max:
            face_region = frame.image[y_min:y_max, x_min:x_max]
            if face_region.any():
                blurred_region = cv2.blur(face_region, (intensity, intensity))
                frame.image[y_min:y_max, x_min:x_max] = blurred_region

    return frame.image
