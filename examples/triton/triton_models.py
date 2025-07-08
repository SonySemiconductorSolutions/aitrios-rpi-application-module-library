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

from typing import List

import numpy as np

from modlib.models.model import COLOR_FORMAT, Model
from modlib.models.post_processors import (
    pp_cls,
    pp_od_bcsn,
    pp_posenet,
    pp_segment,
    pp_anomaly,
)
from modlib.models.results import Classifications, Detections, Poses, Segments, Anomaly


class TritonClassifier(Model):
    def __init__(self):
        super().__init__(
            model_file=None,
            model_type=None,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        from modlib.models.zoo import EfficientNetB0

        self.labels = EfficientNetB0().labels

        # Extra triton requirements
        self.network_file_path = "./examples/assets/triton/classifier/network.fpk"
        self.info_file_path = "./examples/assets/triton/classifier/fpk_info.dat"
        self.input_tensor_shape = (224, 224, 3)
        self.output_tensor_shape_list = [(1000,)]

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        # return pp_cls_softmax(output_tensors)
        return pp_cls(output_tensors)


class TritonDetector(Model):
    def __init__(self):
        super().__init__(
            model_file=None,
            model_type=None,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = ["person", "vest"]

        # Extra triton requirements
        self.network_file_path = "./examples/assets/triton/detector/network.fpk"
        self.info_file_path = "./examples/assets/triton/detector/fpk_info.dat"
        self.input_tensor_shape = (300, 300, 3)
        self.output_tensor_shape_list = [(10, 4), (10,), (10,), (1,)]

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        # return pp_od_bscn(output_tensors)
        return pp_od_bcsn(output_tensors)


class TritonAnomaly(Model):
    def __init__(self):
        super().__init__(
            model_file=None,
            model_type=None,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = ["Anomaly", "Normal"]

        # Extra triton requirements
        self.network_file_path = "./examples/assets/triton/anomaly/network.fpk"
        self.info_file_path = "./examples/assets/triton/anomaly/fpk_info.dat"
        self.input_tensor_shape = (256, 256, 3)
        self.output_tensor_shape_list = [(64, 64, 2)]

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Anomaly:
        return pp_anomaly(output_tensors)


class TritonPoseNet(Model):
    def __init__(self):
        super().__init__(
            model_file=None,
            model_type=None,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

        # Extra triton requirements
        self.network_file_path = "./examples/assets/triton/posenet/network.fpk"
        self.info_file_path = "./examples/assets/triton/posenet/fpk_info.dat"
        self.input_tensor_shape = (481, 353, 3)
        self.output_tensor_shape_list = [(23, 31, 17), (23, 31, 34), (23, 31, 64)]

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Poses:
        return pp_posenet(output_tensors)


class TritonSegmentation(Model):
    def __init__(self):
        super().__init__(
            model_file=None,
            model_type=None,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

        # Extra triton requirements
        self.network_file_path = "./examples/assets/triton/segmentation/network.fpk"
        self.info_file_path = "./examples/assets/triton/segmentation/fpk_info.dat"
        self.input_tensor_shape = (321, 321, 3)
        self.output_tensor_shape_list = [(321, 321)]

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Segments:
        return pp_segment(output_tensors)
