#
# Copyright 2025 Sony Semiconductor Solutions Corp. All rights reserved.
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
import cv2

from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_cls


class YOLOClassification(Model):
    def __init__(self):
        # NOTE: This Sample Code is meant to be used with AI models such as Ultralytics YOLO. Please note that
        # the different license may apply to the AI model you would use and we may not be able to comply
        # your request for source codes except for the Sample Code. For example, Ultralytics YOLO is licensed by
        # Ultralytics Enterprise License, Ultralytics Academic License, or AGPL-3.0 License. If you want
        # to use the Ultralytics YOLO for commercial purpose, you need to purchase the Enterprise License from below.
        # [https://ultralytics.com/license]

        super().__init__(
            model_file="/path/to/yolon_imx_model/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )

        # Default Ultralytics YOLO classifier models trained on ImageNet with 1000 classes, we provide the labels here:
        # ./modlib/models/zoo/assets
        self.labels = np.genfromtxt("/path/to/yolon_imx_model/labels.txt", dtype=str, delimiter="\n")

    def post_process(self, output_tensors):
        return pp_cls(output_tensors)


device = AiCamera()
model = YOLOClassification()
device.deploy(model)

with device as stream:
    for frame in stream:
        for i, label in enumerate([model.labels[id] for id in frame.detections.class_id[:3]]):
            text = f"{i + 1}. {label}: {frame.detections.confidence[i]:.2f}"
            cv2.putText(frame.image, text, (50, 30 + 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)

        frame.display()
