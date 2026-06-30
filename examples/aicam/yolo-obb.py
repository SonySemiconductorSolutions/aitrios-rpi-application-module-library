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

from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model, OBB


class YOLOobb(Model):
    def __init__(self):
        # NOTE: This Sample Code is meant to be used with AI models such as Ultralytics YOLO. Please note that
        # the different license may apply to the AI model you would use and we may not be able to comply
        # your request for source codes except for the Sample Code. For example, Ultralytics YOLO is licensed by
        # Ultralytics Enterprise License, Ultralytics Academic License, or AGPL-3.0 License. If you want
        # to use the Ultralytics YOLO for commercial purpose, you need to purchase the Enterprise License from below.
        # [https://ultralytics.com/license]
        
        super().__init__(
            model_file="/path/to/yolo11n-obb_imx_model/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )

        # DOTA dataset labels
        self.labels = [
            "plane",
            "ship",
            "storage tank",
            "baseball diamond",
            "tennis court",
            "basketball court",
            "ground track field",
            "harbor",
            "bridge",
            "large vehicle",
            "small vehicle",
            "helicopter",
            "roundabout",
            "soccer ball field",
            "swimming pool",
        ]

    def post_process(self, output_tensors):
        input_tensor_sz = 640
        n_detections = int(output_tensors[4][0])
        return OBB(
            bbox = output_tensors[0][:n_detections] / input_tensor_sz,
            class_id = np.array(output_tensors[2][:n_detections], dtype=np.uint8),
            confidence = output_tensors[1][:n_detections],
            angle = output_tensors[3][:n_detections]
        )


device = AiCamera(frame_rate=9)
model = YOLOobb()
device.deploy(model)

annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)


with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.01]
        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _, _ in detections]
        
        annotator.annotate_boxes(frame, detections, labels, alpha=0.3, corner_radius=10)

        frame.display()