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

from typing import Tuple

import cv2
import numpy as np


def yolov8_preprocess(
    x: np.ndarray,
    img_mean: float = 0.0,
    img_std: float = 255.0,
    pad_values: int = 114,
    size: Tuple[int, int] = (640, 640),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess an input image for YOLOv8 model.

    Args:
        x: Input image as a NumPy array.
        img_mean: Mean value used for normalization. Default is 0.0.
        img_std: Standard deviation used for normalization. Default is 255.0.
        pad_values: Value used for padding. Default is 114.
        size: Desired output size (height, width). Default is (640, 640).

    Returns:
        A tuple containing:
        - Preprocessed image as a NumPy array (input_tensor_image).
        - Input tensor ready for model inference.
    """
    h, w = x.shape[:2]  # Image size
    hn, wn = size  # Image new size
    r = max(h / hn, w / wn)
    hr, wr = int(np.round(h / r)), int(np.round(w / r))
    pad = ((int((hn - hr) / 2), int((hn - hr) / 2 + 0.5)), (int((wn - wr) / 2), int((wn - wr) / 2 + 0.5)), (0, 0))

    x = cv2.resize(x, (wr, hr), interpolation=cv2.INTER_AREA)  # Aspect ratio preserving resize
    input_tensor_image = np.pad(x, pad, constant_values=pad_values)  # Padding to the target size

    x = np.flip(input_tensor_image, -1)  # Flip image channels
    x = (x - img_mean) / img_std  # Normalization
    input_tensor = np.expand_dims(x, axis=0)

    return (input_tensor_image, input_tensor)
