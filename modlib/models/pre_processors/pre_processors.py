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

from typing import Tuple, Union, Callable

import cv2
import numpy as np

from modlib.models import COLOR_FORMAT, FRAMEWORK_FORMAT
from modlib.models.results import ROI


def _assert_hwc3(x: np.ndarray) -> None:
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {x.shape}")


def model_normalize(
    x: np.ndarray,
    img_mean: Union[float, np.ndarray] = 0.0,
    img_std: Union[float, np.ndarray] = 255.0,
) -> np.ndarray:
    """Normalize the input tensor to the model image mean and std."""
    # NumPy broadcasting handles scalars and arrays of shape (3,) automatically
    if isinstance(img_mean, np.ndarray):
        if img_mean.shape != (3,):
            raise ValueError(f"img_mean must be a scalar or a numpy array with shape (3,), got shape {img_mean.shape}")

    if isinstance(img_std, np.ndarray):
        if img_std.shape != (3,):
            raise ValueError(f"img_std must be a scalar or a numpy array with shape (3,), got shape {img_std.shape}")

    return (x - img_mean) / img_std


def set_model_color_order(
    x: np.ndarray,
    src_color_format: COLOR_FORMAT,
    model_color_format: COLOR_FORMAT,
) -> np.ndarray:
    """Set color order to the model color format."""
    _assert_hwc3(x)
    if model_color_format == COLOR_FORMAT.RGB and src_color_format == COLOR_FORMAT.BGR:
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    elif model_color_format == COLOR_FORMAT.BGR and src_color_format == COLOR_FORMAT.RGB:
        return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x


def set_model_framework_order(
    x: np.ndarray,
    model_framework_format: FRAMEWORK_FORMAT,
) -> np.ndarray:
    """Set framework order HWC/CHW as required by the model."""
    _assert_hwc3(x)
    if model_framework_format == FRAMEWORK_FORMAT.CHW:
        return np.transpose(x, (2, 0, 1))
    elif model_framework_format == FRAMEWORK_FORMAT.HWC:
        return x


##########################
#### Resize Functions ####
##########################

# NOTE: Optionally define a different resize/cropping/padding strategy with the model pre-process method
#       This function takes in the image as input and should return the resulting image and ROI
#       Any number of additional arguments can be passed to the function as partial arguments
#       E.g. resize_fn=partial(my_custom_function, my_arg=<value>)
ResizeFn = Callable[[np.ndarray], Tuple[np.ndarray, ROI]]


def center_crop(
    x: np.ndarray,
    size: Tuple[int, int],  # (width, height)
) -> Tuple[np.ndarray, ROI]:
    """
    Center crop the image to the given size.

    Args:
        x: Input image
        size: Target size (width, height)

    Returns:
        Tuple containing:
        - Cropped image
        - ROI of the cropped image w.r.t. the original input image.
    """
    h, w = x.shape[:2]
    wn, hn = size
    y1, x1 = (h - hn) // 2, (w - wn) // 2
    x = x[y1 : y1 + hn, x1 : x1 + wn]
    roi = ROI(left=x1 / w, top=y1 / h, width=wn / w, height=hn / h)
    return x, roi


def aspect_ratio_preserving_resize_with_pad(
    x: np.ndarray,
    size: Tuple[int, int],  # (width, height)
    pad_values: int = 114,
) -> Tuple[np.ndarray, ROI]:
    """
    Aspect ratio preserving resize with padding.
    Resizes and pads the image with given `pad_values` to preserving the aspect ratio.

    Args:
        x: Input image
        size: Target size (width, height)
        pad_values: Padding values. Default is 114.

    Returns:
        Tuple containing:
        - Resized and padded image
        - ROI of the resized and padded image w.r.t. the original input image. NOTE: The added padding makes the ROI go outside of the original image, therefore, left and top can be negative and width and height can be greater than 1.
    """

    # Padding & resize
    h, w = x.shape[:2]  # Image size
    wn, hn = size  # Image new size
    r = max(h / hn, w / wn)
    hr, wr = int(np.round(h / r)), int(np.round(w / r))
    pad = ((int((hn - hr) / 2), int((hn - hr) / 2 + 0.5)), (int((wn - wr) / 2), int((wn - wr) / 2 + 0.5)), (0, 0))

    x = cv2.resize(x, (wr, hr), interpolation=cv2.INTER_AREA)  # Aspect ratio preserving resize
    x = np.pad(x, pad, constant_values=pad_values)  # Padding to the target size

    # NOTE: The added padding makes the ROI go outside of the original image,
    # therefore, left and top can be negative and width and height can be greater than 1.
    roi = ROI(left=-pad[1][0] / wr, top=-pad[0][0] / hr, width=wn / wr, height=hn / hr)

    return x, roi


def model_preprocess(
    x: np.ndarray,
    resize_fn: ResizeFn,
    src_color_format: COLOR_FORMAT,
    model_color_format: COLOR_FORMAT,
    norm_mean: Union[float, np.ndarray],
    norm_std: Union[float, np.ndarray],
    framework_format: FRAMEWORK_FORMAT,
) -> Tuple[np.ndarray, np.ndarray, ROI]:
    """
    General model pre-processing function.
    It performs the following steps:
    1. resize/crop/padding. According to the supplied `resize_fn`.
    2. convert to model color format (given the input `src_color_format` and `model_color_format`)
    3. model normalize (given the `norm_mean` and `norm_std`)
    4. set model framework order (given the `framework_format` of the model)

    Args:
        x: Input image
        resize_fn: The resize function to use for the resizing/cropping/padding step
        src_color_format: The color format of the input image
        model_color_format: The color format of the model
        norm_mean: The mean value to use for normalization
        norm_std: The standard deviation value to use for normalization
        framework_format: The framework format of the model

    Returns:
        Tuple containing:
        - input tensor image
        - input tensor
        - ROI of the input tensor w.r.t. the original input image
    """
    # 1. resize/crop/padding.
    # Every model has its own resize/crop/padding strategy, therefore allow flexibility by supplying the right resize_fn.
    # Additionally, it is possible to include ISP limitations by supplying `isp.isp_resize_and_crop(x, model)` as the resize_fn.
    x, roi = resize_fn(x)

    # 2. set model color format # not first ? resize/pad might be influenced by this
    input_tensor_image = set_model_color_order(
        x=x,
        src_color_format=src_color_format,
        model_color_format=model_color_format,
    )

    # 3. model normalize
    input_tensor = model_normalize(input_tensor_image, norm_mean, norm_std)

    # 4. set model framework order
    input_tensor = set_model_framework_order(input_tensor, framework_format)

    return (
        input_tensor_image.astype(np.uint8),
        input_tensor,
        roi,
    )
