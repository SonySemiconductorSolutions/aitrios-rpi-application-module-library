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

from modlib.apps import Annotator
from modlib.devices import Triton

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from triton_models import TritonDetector


def main():
    device = Triton()
    model = TritonDetector()
    device.deploy(model)

    annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

    device.set_image_cropping((0.5, 0.1, 0.5, 0.8))
    device.set_input_tensor_cropping((0.4, 0, 0.5, 0.6))

    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.35]
            labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]

            annotator.annotate_boxes(frame, detections, labels=labels)
            frame.display()


if __name__ == "__main__":
    main()
