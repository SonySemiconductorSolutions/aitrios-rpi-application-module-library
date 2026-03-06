#
# Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.
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

from modlib.models import Model, COLOR_FORMAT
from modlib.models.results import ROI


def _assert_hwc3(x: np.ndarray) -> None:
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"Expected HxWx3, got {x.shape}")


def _assert_3chw(x: np.ndarray) -> None:
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected 3xHxW, got {x.shape}")


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


def isp_resize_and_crop(img: np.ndarray, model: Model) -> Tuple[np.ndarray, ROI]:
    """
    Resize the image to the model's input tensor size.
    No padding is performed. This mimics the functionality of the ISP, which does not include
    extra padding for specific models that require to preserve the aspect ratio.

    Effect of model.preserve_aspect_ratio:
      - If True:
          - Center-crop the image (no padding) to match the aspect ratio
            of the model's input tensor size.
          - Cropping preserves spatial relationships but may result in some edges being lost.
          - Resize to the input tensor size.
          - The returned ROI indicates which region of the original image was used.
      - If False:
          - Resize the entire image to the input tensor size directly,
            ignoring the original aspect ratio.
          - No cropping is performed, and the ROI covers the full image.

    Args:
        img: Input image
        model: Model with given input tensor size and preserve aspect ratio.

    Returns:
        Tuple containing:
        - Resized and cropped image
        - ROI of the resized and cropped image w.r.t. the original input image.
    """

    iw, ih = model.input_tensor_size
    roi = model_aspect_ratio(img, iw, ih, model.preserve_aspect_ratio)

    # Crop image to input tensor aspect ratio using a helper function
    if model.preserve_aspect_ratio:
        img_h, img_w = img.shape[:2]
        img = img[
            int(roi.top * img_h) : int((roi.top + roi.height) * img_h),
            int(roi.left * img_w) : int((roi.left + roi.width) * img_w),
        ]

    # Resize to input tensor size
    r = cv2.resize(img, (iw, ih))

    return r, roi


def isp_normalize_and_quantscale(t: np.ndarray, model: Model) -> np.ndarray:
    """
    Normalization and quantization scaling of the input tensor.
    This function does the equivalent of the following things simultaneously:
    1. Normalization: x_norm = (x - mean) / std
    2. Quantization/scale: q = x_norm * scale + shift

    Args:
        t: Input tensor
        model: Model to normalize and quantscale the input tensor for.

    Returns:
        Normalized and quantized input tensor.
    """
    # 1. Normalization: x_norm = (x - mean) / std
    # 2. Quantization/scale: q = x_norm * scale + shift
    # These normalization and quantization scale operations can be expanded to:
    # q = A * x + B, where A = scale / STD, B = -scale * MEAN / STD + shift

    # And is equivalent to:
    # t = t.astype(np.int32)
    # u = (t * div_val) >> div_shift
    # v = u + norm_val
    # q = v >> norm_shift

    # Because of how the norm_val, norm_shift, div_val, div_shift were computed and stored in the model info.
    # Mathematically, these operations are equial to (note: floor rounding ⌊ ... ⌋):
    # u = ⌊ t * div_val / 2^div_shift ⌋
    # v = u + norm_val
    # q = ⌊ v / 2^norm_shift ⌋

    # Expanding the above:
    # q = ⌊ A' * x + B' ⌋, where A' = div_val / 2^(div_shift + norm_shift), B' = norm_val / 2^norm_shift

    # And the norm_val, norm_shift, div_val, div_shift values are computed from this set:
    # A ≈ A' ≈ scale/STD ≈ div_val / 2^(div_shift + norm_shift)
    # B ≈ B' ≈ -scale * MEAN / STD + shift ≈ norm_val / 2^norm_shift

    norm_val = model.info["input_tensor"]["norm_val"]
    norm_shift = model.info["input_tensor"]["norm_shift"]
    div_val = model.info["input_tensor"]["div_val"]
    div_shift = model.info["input_tensor"]["div_shift"]

    t = t.astype(np.int32)  # uint8 > int32 just for calculations

    # Normalization + quantization/scale on each channel
    # q ≈ (((t * div_val[i]) >> div_shift) + norm_val[i]) >> norm_shift[i]
    # reverse (input tensor denormalization): t ≈ ((((q << norm_shift[i]) - norm_val[i]) << div_shift) // div_val[i]) & 0xFF
    for i in [0, 1, 2]:
        t[i] = (((t[i] * div_val[i]) >> div_shift) + norm_val[i]) >> norm_shift[i]

    # t range now -128-127 or 0-255 depending on dtype
    dtype = model.info["input_tensor"]["dtype"]
    return t.astype(dtype)


def isp_denormalize_input_tensor(t: np.ndarray, model: Model) -> np.ndarray:
    """
    Denormalization of the input tensor.
    This function denormalizes the input tensor of any applied normalization and quantization scaling.
    And is the reverse operation of `isp_normalize_and_quantscale`.

    Args:
        t: Input tensor
        model: Model to denormalize the input tensor for.

    Returns:
        Denormalized input tensor as a image in HWC format.
    """

    norm_val = model.info["input_tensor"]["norm_val"]
    norm_shift = model.info["input_tensor"]["norm_shift"]
    div_val = model.info["input_tensor"]["div_val"]
    div_shift = model.info["input_tensor"]["div_shift"]

    w, h = model.input_tensor_size
    H_pad = h + (h % 2)
    W_pad = w + (32 - (w % 32) if w % 32 != 0 else 0)

    if int(np.asarray(t).size) == 3 * h * w:
        r1 = np.array(t, dtype=np.uint8).astype(np.int32).reshape((3, h, w))  # CHW
    elif int(np.asarray(t).size) == 3 * H_pad * W_pad:
        r1 = np.array(t, dtype=np.uint8).astype(np.int32).reshape((3, H_pad, W_pad))[:, :h, :w]  # CHW
    else:
        raise ValueError(
            f"Unexpected input tensor size: got {int(np.asarray(t).size)} elements, expected {3 * h * w} "
            f"(3x{h}x{w}) or {3 * H_pad * W_pad} (3x{H_pad}x{W_pad})."
        )

    for i in [0, 1, 2]:
        r1[i] = ((((r1[i] << norm_shift[i]) - norm_val[i]) << div_shift) // div_val[i]) & 0xFF

    input_tensor_image = np.transpose(r1, (1, 2, 0)).astype(np.uint8).copy()  # CHW -> HWC
    return input_tensor_image


def isp_padding(t: np.ndarray, model: Model) -> np.ndarray:
    """
    ISP padding for DSP accelerator
    """
    w, h = model.input_tensor_size

    h_pad = h % 2
    w_pad = 32 - (w % 32) if w % 32 != 0 else 0
    padding = [(0, h_pad), (0, w_pad), (0, 0)]
    return np.pad(t, padding, mode="constant", constant_values=0)


def prepare_tensor_like_isp(
    img: np.ndarray,
    model: Model,
    src_color_format: COLOR_FORMAT = COLOR_FORMAT.BGR,  # cv2 imread is always BGR
) -> Tuple[np.ndarray, ROI]:
    """
    Prepare input tensor like IMX500 ISP.

    Args:
        img: The input image
        model: The model with input tensor requirements
        src_color_format: The color format of the input image

    Returns:
        Tuple containing:
        - Input tensor ready to inject into the SDSP accelerator.
        - ROI of the input tensor w.r.t. the original input image.
    """
    # 1. Input tensor resize/crop with ISP limitation on padding
    # CROPS and RESIZES the input image to the right input tensor size
    img, roi = isp_resize_and_crop(img, model)

    # 2. Set model color order
    r = set_model_color_order(x=img, src_color_format=src_color_format, model_color_format=model.color_format)

    # 3. ISP Padding
    r = isp_padding(r, model)

    # 4. Reorder from HWC (from cv2 imread) -> CHW
    _assert_hwc3(r)
    r = np.transpose(r, (2, 0, 1))

    # 5. ISP Normalization
    r = isp_normalize_and_quantscale(r, model)  # range -128-127 or 0-255

    return r.ravel(), roi


def model_aspect_ratio(img: np.ndarray, iw: int, ih: int, preserve_aspect_ratio: bool) -> ROI:
    """
    Get ROI to crop image to input tensor aspect ratio.

    Args:
        img: Input image
        iw: Input tensor width
        ih: Input tensor height
        preserve_aspect_ratio: Whether to preserve the aspect ratio

    Returns:
        normalized ROI relative to the input image.
    """
    if not preserve_aspect_ratio:
        return ROI(left=0, top=0, width=1, height=1)

    # CROP IMAGE TO INPUT TENSOR AR
    img_h, img_w = img.shape[:2]
    model_aspect = iw / ih
    image_aspect = img_w / img_h

    if model_aspect > image_aspect:
        w, h = img_w, int(image_aspect / model_aspect * img_h)
    else:
        w, h = int(model_aspect / image_aspect * img_w), img_h

    x, y = int((img_w - w) / 2), int((img_h - h) / 2)

    # Assert if the computed ROI aspect ratio does not match the model aspect ratio
    roi_aspect = w / h
    # Only check if w and h are nonzero to avoid divide-by-zero
    if w > 0 and h > 0:
        assert abs(roi_aspect - model_aspect) < 1e-2, (
            f"[model_aspect_ratio] ROI w/h ratio ({roi_aspect:.6f}) does not match model aspect ratio ({model_aspect:.6f})"
        )

    return ROI(left=x / img_w, top=y / img_h, width=w / img_w, height=h / img_h)


def extract_scale_and_shift(mean, std, norm_val, norm_shift, div_val, div_shift) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to extract quantization scale and shift from the model info values `norm_val`, `norm_shift`, `div_val` and `div_shift`.
    """
    # A word on the values of norm_val, norm_shift, div_val, div_shift.
    # From `isp_normalize_and_quantscale` it is explained that:
    # q = A * x + B, where A = scale / STD, B = -scale * MEAN / STD + shift
    # q = ⌊ A' * x + B' ⌋, where A' = div_val / 2^(div_shift + norm_shift), B' = norm_val / 2^norm_shift

    # So originally for the operations to be equivalent, the values of norm_val, norm_shift, div_val, div_shift were computed from this set:
    # A ≈ A' ≈ scale/STD = div_val / 2^(div_shift + norm_shift)
    # B ≈ B' ≈ -scale * MEAN / STD + shift ≈ norm_val / 2^norm_shift
    # Since there is no single unique solution to norm_val, norm_shift, div_val, div_shift. Values are computed for IMX500 hardware and
    # Very common a total fixed-point precision is chosen. N = div_shift + norm_shift, e.g. 6 + 4 = 10 bits.

    # Since we have the norm_val, norm_shift, div_val, div_shift available in the rpk file,
    # We can also solve for shift and scale (used during quantization/scale operations, not model normalization)

    # scale = std * div_val / (2 ** (div_shift + norm_shift))
    # shift = norm_val / (2 ** norm_shift) + scale * mean / std

    # This function extracts the scale and shift for each channel

    # Expand the mean and std to 3 channels if needed
    mean = np.asarray(mean)
    if mean.ndim == 0:
        mean = np.full(3, mean)
    std = np.asarray(std)
    if std.ndim == 0:
        std = np.full(3, std)

    # Calculate scale and shift for each channel
    scale, shift = np.zeros(3), np.zeros(3)
    for i in [0, 1, 2]:
        scale[i] = std[i] * div_val[i] / (2 ** (div_shift + norm_shift[i]))
        shift[i] = norm_val[i] / (2 ** norm_shift[i]) + scale[i] * mean[i] / std[i]

    return scale, shift


def prepare_input_tensor_for_dsp(x: np.ndarray, model: Model) -> np.ndarray:
    """
    Prepare input tensor for injection into the DSP accelerator.
    This function assumes the input tensor is already prepared for the framework model and only applies the extra steps specific for the IMX500 DSP.
    Preparing the input tensor by the model preprocessor includes:
        - model color format conversion
        - model resize/padding (with/without preserving aspect ratio)
        - model normalization r = (x - mean) / std
        - set model framework order (HWC/CHW)

    Args:
        x: Input tensor
        model: Model to prepare the input tensor for

    Returns:
        Prepared input tensor ready for injection into the DSP accelerator.
    """

    if not hasattr(model, "norm_mean") or not hasattr(model, "norm_std"):
        raise ValueError("Please define norm_mean and norm_std attributes in the model.")

    # ISP padding
    r = isp_padding(x, model)

    # ISP reorder to CHW
    try:
        _assert_3chw(r)
    except:
        # Asseumed HWC, convert to CHW
        _assert_hwc3(r)
        r = np.transpose(r, (2, 0, 1))

    # Extract scale and shift from model info
    norm_val = model.info["input_tensor"]["norm_val"]
    norm_shift = model.info["input_tensor"]["norm_shift"]
    div_val = model.info["input_tensor"]["div_val"]
    div_shift = model.info["input_tensor"]["div_shift"]
    dtype = model.info["input_tensor"]["dtype"]

    scale, shift = extract_scale_and_shift(model.norm_mean, model.norm_std, norm_val, norm_shift, div_val, div_shift)

    # Quantization r = scale * x + shift
    # Reshape scale and shift to (3, 1, 1) for broadcasting with CHW image
    scale = scale[:, np.newaxis, np.newaxis]
    shift = shift[:, np.newaxis, np.newaxis]
    q = r * scale + shift
    q = np.floor(q).astype(dtype)  # right-shift rounding: floor(x) for x >= 0

    return q.ravel()
