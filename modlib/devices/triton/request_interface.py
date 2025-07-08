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

from enum import Enum
from ctypes import Structure, c_void_p, c_uint32, c_uint16, c_uint8, c_bool, c_double, c_uint64


MAX_NUM_DIMENSIONS = 16
MAX_NUM_TENSORS = 16
REQUEST_POOL_SIZE = 10


# Matches: struct RequestImage
class RequestImage(Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("num_channels", c_uint32),
        ("data_offset", c_uint32),
        ("data_size", c_uint32),
    ]


# Matches: struct RequestInputTensor
class RequestInputTensor(Structure):
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("num_channels", c_uint32),
        ("data_offset", c_uint32),
        ("data_size", c_uint32),
    ]


# Matches: struct OutputTensorInfo
class OutputTensorInfo(Structure):
    _fields_ = [
        ("tensor_data_num", c_uint32),
        ("num_dimensions", c_uint32),
        ("size", c_uint16 * MAX_NUM_DIMENSIONS),
    ]


# Matches: struct RequestOutputTensor
class RequestOutputTensor(Structure):
    _fields_ = [
        ("num_tensors", c_uint32),
        ("info", OutputTensorInfo * MAX_NUM_TENSORS),
        ("data_offset", c_uint32),
        ("data_size", c_uint32),
    ]


# Matches: struct RequestInterface
class RequestInterface(Structure):
    _fields_ = [
        ("idx", c_uint8),
        ("image", RequestImage),
        ("input_tensor", RequestInputTensor),
        ("output_tensor", RequestOutputTensor),
    ]


# Matches: struct RequestPool
class RequestPool(Structure):
    _fields_ = [
        ("in_use", c_bool * REQUEST_POOL_SIZE),
        ("requests", RequestInterface * REQUEST_POOL_SIZE),
        ("current_index", c_uint8),
        ("memory_base", c_void_p),
        ("image_buffer_offset", c_uint32),
        ("input_tensor_offset", c_uint32),
        ("output_tensor_offset", c_uint32),
        ("pool_size", c_uint32),
        ("image_size", c_uint32),
        ("input_tensor_size", c_uint32),
        ("output_tensor_size", c_uint32),
    ]


# Matches: struct TritonConfig
class TritonConfig(Structure):
    _fields_ = [
        ("keep_running", c_bool),
        ("headless", c_bool),
        ("enable_input_tensor", c_bool),
        ("binning_factor", c_uint8),
        ("frame_rate", c_double),
        ("roi_it", c_uint32 * 4),
        ("roi_hires", c_uint32 * 4),
        ("total_pool_size", c_uint64),
    ]


# Matches: enum class CameraFileType
class CameraFileType(Enum):
    """
    Enum for the camera file type
    """

    FILE_DEEP_NEURAL_NETWORK_FIRMWARE = 0  # firmware.fpk
    FILE_DEEP_NEURAL_NETWORK_LOADER = 1  # loader.fpk
    FILE_DEEP_NEURAL_NETWORK_NETWORK = 2  # network.fpk
    FILE_DEEP_NEURAL_NETWORK_INFO = 3  # fpk_info.dat
    FILE_DEEP_NEURAL_NETWORK_CLASSIFICATION = 4  # label.txt
    CAMERA_FILE_TYPE_UNKNOWN = 0xFFFF  # Unknown Camera File Type
