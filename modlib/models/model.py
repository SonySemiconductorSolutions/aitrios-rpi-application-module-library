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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from .results import Anomaly, Classifications, Detections, Poses, Segments, InstanceSegments


@dataclass
class MODEL_TYPE:
    """
    Representation of the available model types corresponding to the provided model file.
    Can be used as e.g. `MODEL_TYPE.KERAS`
    """

    KERAS = "keras"  #: Quantized Keras models (*.keras)
    ONNX = "onnx"  #: Quantized ONNX models (*.onnx)
    CONVERTED = "converted"  #: Converted models by the imx500 converter (*.zip)
    RPK_PACKAGED = "rpk_packaged"  #: Packaged models by the imx500 packager (*.rpk)


@dataclass
class COLOR_FORMAT:
    """
    Representation of the available color formats the provided model is trained in.
    Can be used as e.g. `COLOR_FORMAT.RGB`
    """

    RGB = "RGB"
    BGR = "BGR"


class Model(ABC):
    """
    Abstract base class for models the Application Module Library.

    Can be used as a base class to create custom models.
    When creating a custom model, make sure to always:
    - Initialise the base arguments `model_file`, `model_type`, `color_format` and `preserve_aspect_ratio`
    - Always implement a post_processor function that returns one of the result types.

    For example, a custom classifier could be created like this.
    ```
    from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model, Classifications
    from modlib.models.post_processors import pp_cls

    class CustomClassifier(Model):
        def __init__(self):
            super().__init__(
                model_file="./path/to/network.rpk",
                model_type=MODEL_TYPE.RPK_PACKAGED,
                color_format=COLOR_FORMAT.RGB,
                preserve_aspect_ratio=False,
            )

        def post_process(self, output_tensors) -> Classifications:
            return pp_cls(output_tensors)
    ```
    """

    def __init__(
        self,
        model_file: Path = None,
        model_type: MODEL_TYPE = None,
        color_format: COLOR_FORMAT = COLOR_FORMAT.RGB,
        preserve_aspect_ratio: bool = True,
    ):
        """
        Initialisation of the model base class.

        Args:
            model_file: The path to the model file.
            model_type: The type of the model.
            color_format: The color format of the model.
            preserve_aspect_ratio: Setting the sensor whether or not to preserve aspect ratio of the input tensor.
        """
        self.model_file = os.path.abspath(model_file) if model_file else None
        self.model_type = model_type
        self.color_format = color_format
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self.__info = None

    @abstractmethod
    def post_process(
        self, output_tensors: List[np.ndarray]
    ) -> Union[Classifications, Detections, Poses, Segments, InstanceSegments, Anomaly]:
        """
        Perform post-processing on the tensor data and tensor layout.

        Args:
            output_tensors: Resulting output tensors to be processed.

        Returns:
            The post-processed result.
        """
        pass

    def pre_process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional pre-processing function the model requires for the input image.
        The pre-processing function is mimicking the SPI camera functionality,
        and is only used for data-injection and on Interpreter devices.

        Args:
            image: The input image to be processed.

        Returns:
            A tuple containing:
            - Preprocessed image as a NumPy array (input_tensor_image).
            - Input tensor ready for model inference.
        """
        pass

    @property
    def input_tensor_size(self):
        """
        The input tensor size (width, height) of the model.
        Only available after calling `_get_network_info` on the model's info file.
        """
        if self.__info and "input_tensor" in self.__info:
            return (self.__info["input_tensor"]["width"], self.__info["input_tensor"]["height"])
        else:
            raise ValueError("Input tensor size not available.")

    @property
    def output_tensor_sizes(self):
        """
        The output tensor sizes the model returns.
        Only available after calling `_get_network_info` on the model's info file.
        """
        if self.__info and "output_tensor_sizes" in self.__info:
            return self.__info["output_tensor_sizes"]
        else:
            raise ValueError("Output tensor sizes not available")

    @property
    def info(self):
        """
        Model info. Only available after calling `_get_network_info` on the model's info file.
        """
        if self.__info:
            return self.__info
        else:
            raise ValueError("Model info not available.")

    @info.setter
    def info(self, info):
        self.__info = info

    def _get_network_info(self, info_file: Optional[Path] = None):
        if self.model_type == MODEL_TYPE.RPK_PACKAGED:
            self.__info = self.__get_network_info_rpk(self.model_file)
        elif self.model_type == MODEL_TYPE.CONVERTED and info_file is not None:
            # TODO: Extract model info from the converted packerout file.
            # NOTE: Assumption is that all info is available in packerOut.zip and can be parsed there
            # But for now the info file is only used in the AiCamera passing the rpk file
            # self.__info = self.__get_network_info_converted(self.model_file)
            self.__info = self.__get_network_info_rpk(info_file)
        elif self.model_type in (MODEL_TYPE.KERAS, MODEL_TYPE.ONNX) and info_file is not None:
            # TODO: replace info file by the converted file
            # NOTE: Assumption is that all info is available in packerOut.zip and can be parsed there
            # But for now the info file is only used in the AiCamera passing the rpk file
            self.__info = self.__get_network_info_rpk(info_file)
        else:
            raise ValueError(f"Parsing network info for type {self.model_type}, not supported")

    def __get_network_info_rpk(self, network_filename):
        import io
        import struct

        from libarchive.read import fd_reader

        info = {
            "input_tensor": {
                "width": None,
                "height": None,
                "input_format": None,
                "norm_val": None,
                "norm_shift": None,
                "div_val": None,
                "div_shift": None,
            },
            "network_info": {},
        }

        with open(network_filename, "rb") as fp:
            fw = memoryview(fp.read())
        # Iterate through network firmware discarding blocks
        cpio_offset = 0
        while True:
            # Parse header (+ current block size)
            (magic, size) = struct.unpack(">4sI", fw[:8])
            if not magic == b"9464":
                break

            # Parse flags
            fw = fw[8:]
            flags = struct.unpack("8B", fw[:8])
            device_lock_flag = flags[6]
            fw = fw[(size + 60 - 8) :]  # jump to footer

            # Ensure footer is as expected
            (magic,) = struct.unpack("4s", fw[:4])
            if not magic == b"3695":
                raise RuntimeError(f"No matching footer found in firmware file {network_filename}")
            fw = fw[4:]
            cpio_offset += size + 64

            if (device_lock_flag & 0x01) == 1:
                # skip forward 32 bytes if device_lock_flag.bit0 == 1
                fw = fw[32:]
                cpio_offset += 32

        cpio_fd = os.open(network_filename, os.O_RDONLY)
        os.lseek(cpio_fd, cpio_offset, os.SEEK_SET)
        network_info_raw = None
        with fd_reader(cpio_fd) as archive:
            for entry in archive:
                if "network_info.txt" == str(entry):
                    network_info_raw = b"".join(entry.get_blocks())
        os.close(cpio_fd)
        if network_info_raw is None:
            return
        res = {}
        buf = io.StringIO(network_info_raw.decode("ascii"))
        for line in buf:
            key, value = line.strip().split("=")
            if key == "networkID":
                nid: int = 0
                for idx, x in enumerate(value):
                    nid |= (ord(x) - ord("0")) << (20 - idx * 4)
                res[key] = nid
            if key == "apParamSize":
                res[key] = int(value)
            if key == "networkNum":
                res[key] = int(value)
        res["network"] = {}
        networks = network_info_raw.decode("ascii").split("networkOrdinal=")[1:]
        for nw in networks:
            buf = io.StringIO(nw)
            nw_idx = int(buf.readline())
            nw_properties = {}
            for line in buf:
                key, value = line.strip().split("=")
                nw_properties[key] = value
            res["network"][nw_idx] = nw_properties
        if len(res["network"]) != res["networkNum"]:
            raise RuntimeError("Insufficient networkNum settings in network_info.txt")
        info["network_info"] = res
        # Extract some input tensor config params
        input_format = info["network_info"]["network"][0]["inputTensorFormat"]
        if input_format != self.color_format:
            raise ValueError(
                f"""
                Provided color format ({self.color_format}) does not match the color format found in the packaged rpk file ({input_format}).
                Please apply the correct color format when packaging the model or defining the COLOR_FORMAT of the Model.
                """
            )
        info["input_tensor"]["width"] = int(res["network"][0]["inputTensorWidth"])
        info["input_tensor"]["height"] = int(res["network"][0]["inputTensorHeight"])
        inputTensorNorm_K03 = int(info["network_info"]["network"][0]["inputTensorNorm_K03"], 0)
        inputTensorNorm_K13 = int(info["network_info"]["network"][0]["inputTensorNorm_K13"], 0)
        inputTensorNorm_K23 = int(info["network_info"]["network"][0]["inputTensorNorm_K23"], 0)
        inputTensorNorm_K00 = int(info["network_info"]["network"][0]["inputTensorNorm_K00"], 0)
        inputTensorNorm_K22 = int(info["network_info"]["network"][0]["inputTensorNorm_K22"], 0)
        inputTensorNorm_K02 = int(info["network_info"]["network"][0]["inputTensorNorm_K02"], 0)
        inputTensorNorm_K20 = int(info["network_info"]["network"][0]["inputTensorNorm_K20"], 0)
        inputTensorNorm_K11 = int(info["network_info"]["network"][0]["inputTensorNorm_K11"], 0)
        info["input_tensor"]["input_format"] = input_format
        if input_format == "RGB" or input_format == "BGR":
            norm_val_0 = (
                inputTensorNorm_K03 if ((inputTensorNorm_K03 >> 12) & 1) == 0 else -((~inputTensorNorm_K03 + 1) & 0x1FFF)
            )
            norm_val_1 = (
                inputTensorNorm_K13 if ((inputTensorNorm_K13 >> 12) & 1) == 0 else -((~inputTensorNorm_K13 + 1) & 0x1FFF)
            )
            norm_val_2 = (
                inputTensorNorm_K23 if ((inputTensorNorm_K23 >> 12) & 1) == 0 else -((~inputTensorNorm_K23 + 1) & 0x1FFF)
            )
            norm_val = [norm_val_0, norm_val_1, norm_val_2]
            info["input_tensor"]["norm_val"] = norm_val
            norm_shift = [4, 4, 4]
            info["input_tensor"]["norm_shift"] = norm_shift
            if input_format == "RGB":
                div_val_0 = (
                    inputTensorNorm_K00
                    if ((inputTensorNorm_K00 >> 11) & 1) == 0
                    else -((~inputTensorNorm_K00 + 1) & 0x0FFF)
                )
                div_val_2 = (
                    inputTensorNorm_K22
                    if ((inputTensorNorm_K22 >> 11) & 1) == 0
                    else -((~inputTensorNorm_K22 + 1) & 0x0FFF)
                )
            else:
                div_val_0 = (
                    inputTensorNorm_K02
                    if ((inputTensorNorm_K02 >> 11) & 1) == 0
                    else -((~inputTensorNorm_K02 + 1) & 0x0FFF)
                )
                div_val_2 = (
                    inputTensorNorm_K20
                    if ((inputTensorNorm_K20 >> 11) & 1) == 0
                    else -((~inputTensorNorm_K20 + 1) & 0x0FFF)
                )
            div_val_1 = (
                inputTensorNorm_K11 if ((inputTensorNorm_K11 >> 11) & 1) == 0 else -((~inputTensorNorm_K11 + 1) & 0x0FFF)
            )
            info["input_tensor"]["div_val"] = [div_val_0, div_val_1, div_val_2]
            info["input_tensor"]["div_shift"] = 6

        return info

    @staticmethod
    def __get_network_info_converted(packerOut_file):
        import xml.etree.ElementTree as ET
        import zipfile

        with zipfile.ZipFile(packerOut_file, "r") as zip_file:
            with zip_file.open("dnnParams.xml") as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Select the first network from the networks (networks[0])
                network = root.find(".//network")

                input_tensor = network.find(".//inputTensor")
                input_shape = [int(dim.get("size")) for dim in input_tensor.find("dimensions")]

                output_shapes = []
                for output_tensor in network.findall(".//outputTensor"):
                    shape = [int(dim.get("size")) for dim in output_tensor.find("dimensions")]
                    output_shapes.append(shape)

        # TODO: extract all required information from the packerout file.
        # info = {
        #     "input_tensor": {
        #         "width": None,
        #         "height": None,
        #         "input_format": None,
        #         "norm_val": None,
        #         "norm_shift": None,
        #         "div_val": None,
        #         "div_shift": None,
        #     },
        #     "network_info": {},
        # }

        info = {
            "input_tensor": {
                "width": input_shape[1],
                "height": input_shape[0],
            },
            "output_tensor_sizes": output_shapes,
        }

        return info
