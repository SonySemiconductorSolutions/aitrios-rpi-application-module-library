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

import logging
import re
from typing import Tuple

from .utils import (
    colour_space_from_libcamera,
    colour_space_to_libcamera,
    convert_from_libcamera_type,
    libcamera,
    orientation_to_transform,
    transform_to_orientation,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s]  %(levelname)s %(message)s")
logger = logging.getLogger(__name__.split(".")[-1])


class LibcameraConfig:
    def __init__(
        self,
        camera,
        frame_rate: int,
        enable_input_tensor: bool,
        scaler_crop: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
    ):
        self.camera = camera

        self.camera_config = {
            "use_case": "preview",
            "transform": libcamera.Transform(),
            "colour_space": libcamera.ColorSpace.Sycc(),
            "buffer_count": 28,
            "queue": True,
            "main": {"format": "RGB888", "size": image_size},
            "lores": None,
            "raw": {"format": "SRGGB10_CSI2P", "size": (2028, 1520)},  # Best effort of 2 available modes
            "controls": self.get_controls(
                frame_rate=frame_rate, enable_input_tensor=enable_input_tensor, scaler_crop=scaler_crop
            ),
            "sensor": {},
            "display": "main",
            "encode": "main",
        }

        self.main_index = 0
        self.lores_index = -1
        self.raw_index = -1

        self.libcamera_config = None

    @property
    def camera_controls(self):
        return {
            k.name: (
                convert_from_libcamera_type(v.min),
                convert_from_libcamera_type(v.max),
                convert_from_libcamera_type(v.default),
            )
            for k, v in self.camera.controls.items()
        }

    def get_controls(self, frame_rate: int, enable_input_tensor: bool, scaler_crop: Tuple[int, int, int, int]) -> dict:
        controls = {"FrameRate": frame_rate, "CnnEnableInputTensor": enable_input_tensor, "ScalerCrop": scaler_crop}

        if "NoiseReductionMode" in self.camera_controls and "FrameDurationLimits" in self.camera_controls:
            controls = {
                "NoiseReductionMode": libcamera.controls.draft.NoiseReductionModeEnum.Minimal,
                "FrameDurationLimits": (100, 83333),
            } | controls

        return controls

    @property
    def streams(self):
        if self.libcamera_config is None:
            raise RuntimeError("libcamera_config does not exist. Call create() on the LibcameraConfig object.")

        return [stream_config.stream for stream_config in self.libcamera_config]

    @property
    def libcamera_controls(self):
        if self.libcamera_config is None:
            raise RuntimeError("libcamera_config does not exist. Call create() on the LibcameraConfig object.")

        camera_ctrl_info = {k.name: (k, v) for k, v in self.camera.controls.items()}

        tmp = {}
        tmp_k = []

        def _framerates_to_durations_(framerates):
            if not isinstance(framerates, (tuple, list)):
                framerates = (framerates, framerates)
            return (int(1000000 / framerates[1]), int(1000000 / framerates[0]))

        def _durations_to_framerates_(durations):
            if durations[0] == durations[1]:
                return 1000000 / durations[0]
            return (1000000 / durations[1], 1000000 / durations[0])

        _VIRTUAL_FIELDS_MAP_ = {
            "FrameRate": ("FrameDurationLimits", _framerates_to_durations_, _durations_to_framerates_)
        }

        for k, v in self.camera_config["controls"].items():
            if not k.startswith("_"):
                if k in _VIRTUAL_FIELDS_MAP_:
                    real_field = _VIRTUAL_FIELDS_MAP_[k]
                    k = real_field[0]
                    v = real_field[1](v)
                if k not in camera_ctrl_info.keys():
                    raise RuntimeError(f"Control {k} is not advertised by libcamera")
                tmp_k.append(k)
            tmp[k] = v

        def list_or_tuple(thing):
            return type(thing) in {list, tuple}

        libcamera_controls = {}
        for k in tmp_k:
            v = tmp[k]
            id = camera_ctrl_info[k][0]
            if id.type == libcamera.ControlType.Rectangle:
                # We can get a list of Rectangles or a single one.
                if list_or_tuple(v) and v and list_or_tuple(v[0]):
                    v = [libcamera.Rectangle(*i) for i in v]
                else:
                    v = libcamera.Rectangle(*v)
            elif id.type == libcamera.ControlType.Size:
                v = libcamera.Size(*v)
            libcamera_controls[id] = v

        return libcamera_controls

    def get_raw_modes(self) -> list:
        raw_config = self.camera.generate_configuration([libcamera.StreamRole.Raw])
        raw_formats = raw_config.at(0).formats
        raw_modes = []
        for pix in raw_formats.pixel_formats:
            raw_modes += [{"format": str(pix), "size": (size.width, size.height)} for size in raw_formats.sizes(pix)]
        return raw_modes

    def create(self):
        # Make a libcamera configuration object from our Python configuration.

        # We will create each stream with the "viewfinder" role just to get the stream
        # configuration objects, and note the positions our named streams will have in
        # libcamera's stream list.
        camera_config = self.camera_config.copy()

        roles = [libcamera.StreamRole.Viewfinder]
        index = 1
        if camera_config["lores"] is not None:
            self.lores_index = index
            index += 1
            roles += [libcamera.StreamRole.Viewfinder]
        if camera_config["raw"] is not None:
            self.raw_index = index
            roles += [libcamera.StreamRole.Raw]

        # Create libcamera config
        libcamera_config = self.camera.generate_configuration(roles)
        libcamera_config.orientation = transform_to_orientation(camera_config["transform"])
        buffer_count = camera_config["buffer_count"]
        self._update_libcamera_stream_config(libcamera_config.at(self.main_index), camera_config["main"], buffer_count)
        libcamera_config.at(self.main_index).color_space = colour_space_to_libcamera(
            camera_config["colour_space"], camera_config["main"]["format"]
        )
        if self.lores_index >= 0:
            self._update_libcamera_stream_config(
                libcamera_config.at(self.lores_index), camera_config["lores"], buffer_count
            )
            # Must be YUV, so no need for colour_space_to_libcamera.
            libcamera_config.at(self.lores_index).color_space = camera_config["colour_space"]
        if self.raw_index >= 0:
            self._update_libcamera_stream_config(libcamera_config.at(self.raw_index), camera_config["raw"], buffer_count)
            libcamera_config.at(self.raw_index).color_space = libcamera.ColorSpace.Raw()

        # if not self._is_rpi_camera():
        #     return libcamera_config

        bit_depth = self._get_bit_depth(camera_config["raw"]["format"])
        output_size = camera_config["raw"]["size"]

        # Now find a camera mode that best matches these, and that's what we use.
        # This function copies how libcamera scores modes:
        def score_mode(mode, bit_depth, output_size):
            mode_bit_depth = self._get_bit_depth(mode["format"])
            mode_output_size = mode["size"]
            ar = output_size[0] / output_size[1]
            mode_ar = mode_output_size[0] / mode_output_size[1]

            def score_format(desired, actual):
                score = desired - actual
                return -score / 4 if score < 0 else score * 2

            score = score_format(output_size[0], mode_output_size[0])
            score += score_format(output_size[1], mode_output_size[1])
            score += 1500 * score_format(ar, mode_ar)
            score += 500 * abs(bit_depth - mode_bit_depth)
            return score

        mode = min(self.get_raw_modes(), key=lambda x: score_mode(x, bit_depth, output_size))
        libcamera_config.sensor_config = libcamera.SensorConfiguration()
        libcamera_config.sensor_config.bit_depth = self._get_bit_depth(mode["format"])
        libcamera_config.sensor_config.output_size = libcamera.Size(*mode["size"])

        # VALIDATE: Check that libcamera is happy with it.
        status = libcamera_config.validate()
        self._update_camera_config(camera_config, libcamera_config)
        logger.debug(f"Requesting configuration: {camera_config}")
        if status == libcamera.CameraConfiguration.Status.Invalid:
            raise RuntimeError(f"Invalid camera configuration: {camera_config}")
        elif status == libcamera.CameraConfiguration.Status.Adjusted:
            logger.info("Camera configuration has been adjusted!")

        # Save possibly changed camera config and libcamera config
        self.camera_config = camera_config
        self.libcamera_config = libcamera_config

        return libcamera_config

    # Utilities for creating libcamera configuration
    @staticmethod
    def _update_libcamera_stream_config(libcamera_stream_config, stream_config, buffer_count) -> None:
        # Update the libcamera stream config with ours.
        libcamera_stream_config.size = libcamera.Size(stream_config["size"][0], stream_config["size"][1])
        libcamera_stream_config.pixel_format = libcamera.PixelFormat(stream_config["format"])
        libcamera_stream_config.buffer_count = buffer_count
        # Stride is sometimes set to None in the stream_config, so need to guard against that case
        if stream_config.get("stride") is not None:
            libcamera_stream_config.stride = stream_config["stride"]
        else:
            libcamera_stream_config.stride = 0

    @staticmethod
    def _get_bit_depth(sensor_format_str):
        if "_" in sensor_format_str:
            pixels, packing = sensor_format_str.split("_", 1)
        else:
            pixels, packing = sensor_format_str, None

        bit_depth = 16 if packing in ["PISP_COMP1", "PISP_COMP2"] else int(re.search("\\d+$", pixels).group())
        return bit_depth

    def _update_camera_config(self, camera_config, libcamera_config) -> None:
        camera_config["transform"] = orientation_to_transform(libcamera_config.orientation)
        camera_config["colour_space"] = colour_space_from_libcamera(libcamera_config.at(0).color_space)

        def _update_stream_config(stream_config, libcamera_stream_config) -> None:
            # Update our stream config from libcamera's.
            stream_config["format"] = str(libcamera_stream_config.pixel_format)
            stream_config["size"] = (libcamera_stream_config.size.width, libcamera_stream_config.size.height)
            stream_config["stride"] = libcamera_stream_config.stride
            stream_config["framesize"] = libcamera_stream_config.frame_size

        _update_stream_config(camera_config["main"], libcamera_config.at(0))
        if self.lores_index >= 0:
            _update_stream_config(camera_config["lores"], libcamera_config.at(self.lores_index))
        if self.raw_index >= 0:
            _update_stream_config(camera_config["raw"], libcamera_config.at(self.raw_index))

        if libcamera_config.sensor_config is not None:
            sensor_config = {}
            sensor_config["bit_depth"] = libcamera_config.sensor_config.bit_depth
            sensor_config["output_size"] = convert_from_libcamera_type(libcamera_config.sensor_config.output_size)
            camera_config["sensor"] = sensor_config
