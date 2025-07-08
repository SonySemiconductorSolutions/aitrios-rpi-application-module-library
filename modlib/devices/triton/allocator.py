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

import os
import math
import mmap
import platform
import ctypes
from typing import Tuple, List

from .request_interface import TritonConfig, RequestPool, REQUEST_POOL_SIZE

SENSOR_WIDTH = 4052
SENSOR_HEIGHT = 3036


class ConfigAllocator:
    def __init__(self):
        self._config_mmap = None
        self._config = None
        self.total_size = ctypes.sizeof(TritonConfig)

        self._running = None

    def allocate(self) -> int:
        if platform.system() == "Windows":
            # Create Windows named shared memory
            self._config_mmap = mmap.mmap(-1, self.total_size, tagname="Local\\TritonConfig")
        elif platform.system() == "Linux":
            # Create Linux named shared memory
            self._shm_fd = os.open("/dev/shm/TritonConfig", os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(self._shm_fd, self.total_size)
            self._config_mmap = mmap.mmap(
                self._shm_fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ
            )
        else:
            raise ValueError("Unsupported platform")
        self._config = TritonConfig.from_buffer(self._config_mmap)

        # Initialize with default values
        self._running = False

        self._config.keep_running = False
        self._config.headless = False
        self._config.enable_input_tensor = False
        self._config.binning_factor = 1
        self._config.frame_rate = 30.0
        self._config.roi_it = (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)
        self._config.roi_hires = (0, 0, SENSOR_WIDTH, SENSOR_HEIGHT)

    def cleanup(self):
        """Clean up allocated shared memory."""
        if self._config:
            self._config = None

        if self._config_mmap:
            self._config_mmap.close()
            self._config_mmap = None

        if hasattr(self, "_shm_fd"):
            os.close(self._shm_fd)
            self._shm_fd = None
            if os.path.exists("/dev/shm/TritonConfig"):
                os.unlink("/dev/shm/TritonConfig")

    def get_config(self) -> TritonConfig:
        """Get the config structure from shared memory."""
        if self._config is None:
            raise ValueError("Config is not allocated")
        return self._config

    def set_config(self, config: TritonConfig):
        """Update the config structure in shared memory."""
        if self._config is None:
            raise ValueError("Config is not allocated")
        ctypes.memmove(ctypes.addressof(self._config), ctypes.addressof(config), ctypes.sizeof(TritonConfig))

    def set_keep_running(self, keep_running: bool):
        if self._config is None:
            raise ValueError("Config is not allocated")
        self._running = keep_running
        self._config.keep_running = keep_running


class Allocator:
    def __init__(self):
        self._request_pool_mmap = None
        self._request_pool = None

        self.total_size = None

    def allocate(
        self,
        image_shape: Tuple[int, int, int],
        input_tensor_shape: Tuple[int, int, int],
        output_tensor_shape_list: List[Tuple[int, ...]],
    ) -> int:
        # TODO: check shapes are valid

        # Calculate total size needed
        pool_size = ctypes.sizeof(RequestPool)

        image_size = math.prod(image_shape) * ctypes.sizeof(ctypes.c_uint8)
        input_tensor_size = math.prod(input_tensor_shape) * ctypes.sizeof(ctypes.c_uint8)
        output_tensor_size = sum(
            math.prod(t_shape) * ctypes.sizeof(ctypes.c_float) for t_shape in output_tensor_shape_list
        )

        # Offsets
        image_buffer_offset = pool_size
        input_tensor_offset = pool_size + (image_size) * REQUEST_POOL_SIZE
        output_tensor_offset = pool_size + (image_size + input_tensor_size) * REQUEST_POOL_SIZE
        self.total_size = pool_size + (image_size + input_tensor_size + output_tensor_size) * REQUEST_POOL_SIZE

        if platform.system() == "Windows":
            self._request_pool_mmap = mmap.mmap(-1, self.total_size, tagname="Local\\TritonRequestPool")
        elif platform.system() == "Linux":
            self._shm_fd = os.open("/dev/shm/TritonRequestPool", os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(self._shm_fd, self.total_size)
            self._request_pool_mmap = mmap.mmap(
                self._shm_fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ
            )
        else:
            raise ValueError("Unsupported platform")

        self._request_pool = RequestPool.from_buffer(self._request_pool_mmap)

        # Initiate the offsets
        # self._request_pool.memory_base = ctypes.addressof(ctypes.c_char.from_buffer(self._request_pool_mmap))
        self._request_pool.image_buffer_offset = image_buffer_offset
        self._request_pool.input_tensor_offset = input_tensor_offset
        self._request_pool.output_tensor_offset = output_tensor_offset

        self._request_pool.pool_size = pool_size
        self._request_pool.image_size = image_size
        self._request_pool.input_tensor_size = input_tensor_size
        self._request_pool.output_tensor_size = output_tensor_size

    def cleanup(self):
        """Clean up allocated shared memory."""
        if self._request_pool:
            self._request_pool = None

        if self._request_pool_mmap:
            self._request_pool_mmap.close()
            self._request_pool_mmap = None

        if hasattr(self, "_shm_fd"):
            os.close(self._shm_fd)
            self._shm_fd = None
            if os.path.exists("/dev/shm/TritonRequestPool"):
                os.unlink("/dev/shm/TritonRequestPool")

    def mmap(self):
        if self._request_pool_mmap is None:
            raise ValueError("Request pool mmap is not allocated")
        return self._request_pool_mmap

    def release_request(self, idx: int):
        if not (0 <= idx < REQUEST_POOL_SIZE):
            raise IndexError(f"Index {idx} out of range (0..{REQUEST_POOL_SIZE - 1})")

        self._request_pool.in_use[idx] = False

    def get_request(self, idx: int):
        if not (0 <= idx < REQUEST_POOL_SIZE):
            raise IndexError(f"Index {idx} out of range (0..{REQUEST_POOL_SIZE - 1})")

        return self._request_pool.requests[idx]
