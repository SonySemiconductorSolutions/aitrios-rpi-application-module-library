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

import importlib.resources
import numpy as np

from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics


class YOLO(Model):
    def __init__(self):
        
        # NOTE: This Sample Code is meant to be used with AI models such as Ultralytics YOLO. Please note that
        # the different license may apply to the AI model you would use and we may not be able to comply 
        # your request for source codes except for the Sample Code. For example, Ultralytics YOLO is licensed by
        # Ultralytics Enterprise License, Ultralytics Academic License, or AGPL-3.0 License. If you want
        # to use the Ultralytics YOLO for commercial purpose, you need to purchase the Enterprise License from below.
        # [https://ultralytics.com/license]
        
        super().__init__(
            model_file="/path/to/yolo11n_imx_model/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )

        self.labels = np.genfromtxt("/path/to/yolo11n_imx_model/labels.txt",
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)


device = AiCamera(frame_rate=16)  # Optimal frame rate for maximum DPS of the YOLO model running on the AI Camera
model = YOLO()
device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]
        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]

        annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
        frame.display()
