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

import numpy as np
import pytest

import modlib.devices.imx500.isp as isp
from modlib.models import COLOR_FORMAT
from modlib.models.results import ROI


class MockModel:
    def __init__(self, width, height,
        norm_val, norm_shift, div_val, div_shift, dtype,
        preserve_aspect_ratio=True,
        color_format=COLOR_FORMAT.RGB,
        norm_mean=None, norm_std=None,
    ):
        self.width = width
        self.height = height
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.color_format = color_format
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self._info = {
            "input_tensor": {
                "norm_val": np.array(norm_val, dtype=np.int32),
                "norm_shift": np.array(norm_shift, dtype=np.int32),
                "div_val": np.array(div_val, dtype=np.int32),
                "div_shift": div_shift,
                "dtype": dtype,
            }
        }

    @property
    def input_tensor_size(self):
        return (self.width, self.height)

    @property
    def info(self):
        return self._info


@pytest.fixture
def mock_model_int8():
    """Mock model with int8 dtype (signed), A configuration like SSD."""
    return MockModel(
        width=224,
        height=224,
        norm_val=[-2048, -2048, -2048],
        norm_shift=[4, 4, 4],
        div_val=[1024, 1024, 1024],
        div_shift=6,
        dtype=np.int8,
    )


@pytest.fixture
def mock_model_uint8():
    """Mock model with uint8 dtype (unsigned), A configuration like YOLO."""
    return MockModel(
        width=640,
        height=640,
        norm_val=[0, 0, 0],
        norm_shift=[4, 4, 4],
        div_val=[1024, 1024, 1024],
        div_shift=6,
        dtype=np.uint8,
    )


def test_isp_normalize_and_quantscale_shape(mock_model_int8):
    m = mock_model_int8
    np.random.seed(42)

    # Input for normalization and quantization scaling is expected to be in CHW format
    sample_image_chw = np.random.randint(0, 256, size=(3, m.height, m.width), dtype=np.uint8) # CHW

    result = isp.isp_normalize_and_quantscale(sample_image_chw, m)
    assert result.shape == sample_image_chw.shape
    assert result.dtype == m.info["input_tensor"]["dtype"]


def test_isp_denormalize_input_tensor_shape(mock_model_int8):
    m = mock_model_int8
    np.random.seed(666)

    # Input for normalization and quantization scaling is expected to be in CHW format
    sample_image_chw = np.random.randint(0, 256, size=(3, m.height, m.width), dtype=np.uint8) # CHW

    # Normalize
    normalized = isp.isp_normalize_and_quantscale(sample_image_chw, m)

    flattened = normalized.ravel()
    result = isp.isp_denormalize_input_tensor(flattened, m)
    assert result.shape == (m.height, m.width, 3)
    assert result.dtype == np.uint8


def test_round_trip_int8(mock_model_int8):
    np.random.seed(123)
    sample_image = np.random.randint(0, 256, size=(mock_model_int8.height, mock_model_int8.width, 3), dtype=np.uint8)

    # Normalize
    sample_image_chw = np.transpose(sample_image, (2, 0, 1)) # convert to CHW
    normalized = isp.isp_normalize_and_quantscale(sample_image_chw, mock_model_int8)
    assert normalized.shape == sample_image_chw.shape
    assert normalized.dtype == mock_model_int8.info["input_tensor"]["dtype"]

    # Denormalize
    flattened = normalized.ravel()
    denormalized = isp.isp_denormalize_input_tensor(flattened, mock_model_int8)
    assert denormalized.shape == sample_image.shape
    assert denormalized.dtype == np.uint8

    # Assert that the original image and the denormalized image are the same
    assert np.all(sample_image == denormalized)


def test_round_trip_uint8(mock_model_uint8):
    np.random.seed(456)
    sample_image = np.random.randint(0, 256, size=(mock_model_uint8.height, mock_model_uint8.width, 3), dtype=np.uint8)

    # Normalize
    sample_image_chw = np.transpose(sample_image, (2, 0, 1)) # convert to CHW
    normalized = isp.isp_normalize_and_quantscale(sample_image_chw, mock_model_uint8)
    assert normalized.shape == sample_image_chw.shape
    assert normalized.dtype == np.uint8

    # Denormalize
    flattened = normalized.ravel()
    denormalized = isp.isp_denormalize_input_tensor(flattened, mock_model_uint8)
    assert denormalized.shape == sample_image.shape
    assert denormalized.dtype == np.uint8

    # Assert that the original image and the denormalized image are the same
    assert np.all(sample_image == denormalized)


# ============================================================================
# Tests for set_model_color_order
# ============================================================================

def test_set_model_color_order_bgr_to_rgb():
    # Create a BGR image: blue pixel at (0,0), green at (0,1), red at (0,2)
    bgr_image = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr_image[0, 0] = [255, 0, 0]  # Blue in BGR
    bgr_image[0, 1] = [0, 255, 0]  # Green in BGR
    bgr_image[0, 2] = [0, 0, 255]  # Red in BGR

    rgb_image = isp.set_model_color_order(bgr_image, COLOR_FORMAT.BGR, COLOR_FORMAT.RGB)

    # Verify channel order changed
    assert rgb_image[0, 0, 0] == 0  # R channel should be 0 (was B)
    assert rgb_image[0, 0, 2] == 255  # B channel should be 255 (was R)
    assert rgb_image[0, 1, 1] == 255  # G channel unchanged
    assert rgb_image.shape == bgr_image.shape
    assert rgb_image.dtype == bgr_image.dtype


def test_set_model_color_order_rgb_to_bgr():
    # Create an RGB image
    rgb_image = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_image[0, 0] = [255, 0, 0]  # Red in RGB
    rgb_image[0, 1] = [0, 255, 0]  # Green in RGB
    rgb_image[0, 2] = [0, 0, 255]  # Blue in RGB

    bgr_image = isp.set_model_color_order(rgb_image, COLOR_FORMAT.RGB, COLOR_FORMAT.BGR)

    # Verify channel order changed
    assert bgr_image[0, 0, 2] == 255  # B channel should be 255 (was R)
    assert bgr_image[0, 0, 0] == 0  # R channel should be 0 (was B)
    assert bgr_image[0, 1, 1] == 255  # G channel unchanged
    assert bgr_image.shape == rgb_image.shape
    assert bgr_image.dtype == rgb_image.dtype


def test_set_model_color_order_same_format():
    image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    original = image.copy()

    # BGR to BGR
    result_bgr = isp.set_model_color_order(image, COLOR_FORMAT.BGR, COLOR_FORMAT.BGR)
    assert np.array_equal(result_bgr, original)
    assert result_bgr is image  # Should return same object

    # RGB to RGB
    result_rgb = isp.set_model_color_order(image, COLOR_FORMAT.RGB, COLOR_FORMAT.RGB)
    assert np.array_equal(result_rgb, original)
    assert result_rgb is image  # Should return same object


def test_set_model_color_order_invalid_shape():
    # 2D array
    invalid_2d = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected HxWx3"):
        isp.set_model_color_order(invalid_2d, COLOR_FORMAT.BGR, COLOR_FORMAT.RGB)


# ============================================================================
# Tests for isp_resize_and_crop
# ============================================================================

def test_isp_resize_and_crop_preserve_aspect_ratio_false():
    """Test isp_resize_and_crop with preserve_aspect_ratio=False."""
    model = MockModel(
        width=224,
        height=224,
        norm_val=[0, 0, 0],
        norm_shift=[4, 4, 4],
        div_val=[1024, 1024, 1024],
        div_shift=6,
        dtype=np.uint8,
        preserve_aspect_ratio=False,
    )

    # Create a wide image (640x480)
    img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)

    resized, roi = isp.isp_resize_and_crop(img, model)

    # Should be resized to model size
    assert resized.shape == (224, 224, 3)
    # ROI should cover full image
    assert roi.left == 0.0
    assert roi.top == 0.0
    assert roi.width == 1.0
    assert roi.height == 1.0


def test_isp_resize_and_crop_preserve_aspect_ratio_true_wider_image():
    """Test isp_resize_and_crop with preserve_aspect_ratio=True, image wider than model."""
    model = MockModel(
        width=224,  # Square model (1:1)
        height=224,
        norm_val=[0, 0, 0],
        norm_shift=[4, 4, 4],
        div_val=[1024, 1024, 1024],
        div_shift=6,
        dtype=np.uint8,
        preserve_aspect_ratio=True,
    )

    # Create a wide image (640x480, aspect ratio 4:3)
    # Model is 1:1, image is 4:3, so model is taller -> crop width
    img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)

    resized, roi = isp.isp_resize_and_crop(img, model)

    # Should be resized to model size
    assert resized.shape == (224, 224, 3)
    # ROI should be cropped (width cropped, height full) because model is taller
    assert roi.top == 0.0
    assert roi.height == 1.0
    assert roi.width < 1.0  # Width was cropped
    assert roi.left > 0.0  # Cropped from left
    # Verify ROI values are normalized
    assert 0.0 <= roi.left <= 1.0
    assert 0.0 <= roi.top <= 1.0
    assert 0.0 <= roi.width <= 1.0
    assert 0.0 <= roi.height <= 1.0


def test_isp_resize_and_crop_preserve_aspect_ratio_true_taller_image():
    """Test isp_resize_and_crop with preserve_aspect_ratio=True, image taller than model."""
    model = MockModel(
        width=320,  # Wide model (4:3 aspect ratio)
        height=240,
        norm_val=[0, 0, 0],
        norm_shift=[4, 4, 4],
        div_val=[1024, 1024, 1024],
        div_shift=6,
        dtype=np.uint8,
        preserve_aspect_ratio=True,
    )

    # Create a tall image (640x480, aspect ratio 3:4)
    # Model is 4:3, image is 3:4, so model is wider -> crop height
    img = np.random.randint(0, 256, size=(640, 480, 3), dtype=np.uint8)

    resized, roi = isp.isp_resize_and_crop(img, model)

    # Should be resized to model size
    assert resized.shape == (240, 320, 3)
    # ROI should be cropped (height cropped, width full) because model is wider
    assert roi.left == 0.0
    assert roi.width == 1.0
    assert roi.height < 1.0  # Height was cropped
    assert roi.top > 0.0  # Cropped from top


# ============================================================================
# Tests for isp_padding
# ============================================================================

def test_isp_padding_even_height():
    """Test isp_padding with even height (no height padding)."""
    model = MockModel(
        width=224,
        height=224,  # Even height
        norm_val=[0, 0, 0],
        norm_shift=[4, 4, 4],
        div_val=[1024, 1024, 1024],
        div_shift=6,
        dtype=np.uint8,
    )

    img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    padded = isp.isp_padding(img, model)

    # Height should not change (224 % 2 == 0)
    assert padded.shape[0] == 224
    # Width padding: 224 % 32 == 0, so no padding
    assert padded.shape[1] == 224
    assert padded.shape[2] == 3


def test_isp_padding_odd_height():
    """Test isp_padding with odd height (add 1 row)."""
    model = MockModel(
        width=100,  # 100 % 32 = 4, so will be padded to 128
        height=225,  # Odd height
        norm_val=[0, 0, 0],
        norm_shift=[4, 4, 4],
        div_val=[1024, 1024, 1024],
        div_shift=6,
        dtype=np.uint8,
    )

    img = np.random.randint(0, 256, size=(225, 100, 3), dtype=np.uint8)
    padded = isp.isp_padding(img, model)

    # Height should be padded by 1 (225 + 1 = 226)
    assert padded.shape[0] == 226
    # Width should be padded to next multiple of 32 (128)
    assert padded.shape[1] == 128
    assert padded.shape[2] == 3
    # Verify padding values are 0
    assert np.all(padded[225:, :, :] == 0)
    assert np.all(padded[:, 100:, :] == 0)


# ============================================================================
# Tests for model_aspect_ratio
# ============================================================================

def test_model_aspect_ratio_preserve_true_wider_model():
    """Test model_aspect_ratio with preserve_aspect_ratio=True, model wider than image."""
    # Model: 320x240 (4:3 aspect ratio)
    # Image: 480x480 (1:1 aspect ratio)
    img = np.random.randint(0, 256, size=(480, 480, 3), dtype=np.uint8)
    roi = isp.model_aspect_ratio(img, 320, 240, preserve_aspect_ratio=True)

    # Should crop height (center crop)
    assert roi.left == 0.0
    assert roi.width == 1.0
    assert roi.height < 1.0
    assert roi.top > 0.0
    # Verify ROI aspect ratio matches model
    img_h, img_w = img.shape[:2]
    roi_w = roi.width * img_w
    roi_h = roi.height * img_h
    model_aspect = 320 / 240
    roi_aspect = roi_w / roi_h
    assert abs(roi_aspect - model_aspect) < 1e-2


def test_model_aspect_ratio_preserve_true_taller_model():
    """Test model_aspect_ratio with preserve_aspect_ratio=True, model taller than image."""
    # Model: 240x320 (3:4 aspect ratio)
    # Image: 640x480 (4:3 aspect ratio)
    img = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    roi = isp.model_aspect_ratio(img, 240, 320, preserve_aspect_ratio=True)

    # Should crop width (center crop)
    assert roi.top == 0.0
    assert roi.height == 1.0
    assert roi.width < 1.0
    assert roi.left > 0.0
    # Verify ROI aspect ratio matches model
    img_h, img_w = img.shape[:2]
    roi_w = roi.width * img_w
    roi_h = roi.height * img_h
    model_aspect = 240 / 320
    roi_aspect = roi_w / roi_h
    assert abs(roi_aspect - model_aspect) < 1e-2


def test_model_aspect_ratio_preserve_true_matching():
    """Test model_aspect_ratio with matching aspect ratio."""
    # Model: 224x224 (1:1)
    # Image: 480x480 (1:1)
    img = np.random.randint(0, 256, size=(480, 480, 3), dtype=np.uint8)
    roi = isp.model_aspect_ratio(img, 224, 224, preserve_aspect_ratio=True)

    # Should cover full image
    assert roi.left == 0.0
    assert roi.top == 0.0
    assert roi.width == 1.0
    assert roi.height == 1.0


# ============================================================================
# Tests for extract_scale_and_shift
# ============================================================================

def test_extract_scale_and_shift_calculation():
    """Test extract_scale_and_shift calculation correctness."""
    mean = 128.0
    std = 64.0
    norm_val = np.array([100, 100, 100], dtype=np.int32)
    norm_shift = np.array([4, 4, 4], dtype=np.int32)
    div_val = np.array([1024, 1024, 1024], dtype=np.int32)
    div_shift = 6

    scale, shift = isp.extract_scale_and_shift(mean, std, norm_val, norm_shift, div_val, div_shift)

    # Verify scale calculation: scale = std * div_val / (2 ** (div_shift + norm_shift))
    expected_scale = 64.0 * 1024 / (2 ** (6 + 4))
    assert np.allclose(scale, expected_scale)

    # Verify shift calculation: shift = norm_val / (2 ** norm_shift) + scale * mean / std
    expected_shift = 100 / (2 ** 4) + expected_scale * 128 / 64
    assert np.allclose(shift, expected_shift)
