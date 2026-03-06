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

from modlib.apps import Annotator
from modlib.devices import Video, InterpreterClient
from modlib.models.zoo import YOLO11n


# NOTE: The InterpreterClient requires an inference server running simultaneously.
# You can find a sample Docker-based inference server in ./examples/interpreters/docker.
# Make sure to check the README in that folder for instructions on installing and launching the server.
device = InterpreterClient(
    source=Video("./examples/assets/palace.mp4"),
    endpoint="http://localhost:8000",
    enable_input_tensor=False,
)

# NOTE: This Sample Code is meant to be used with AI models such as Ultralytics YOLO. Please note that
# the different license may apply to the AI model you would use and we may not be able to comply
# your request for source codes except for the Sample Code. For example, Ultralytics YOLO is licensed by
# Ultralytics Enterprise License, Ultralytics Academic License, or AGPL-3.0 License. If you want
# to use the Ultralytics YOLO for commercial purpose, you need to purchase the Enterprise License from below.
# [https://ultralytics.com/license]
model = YOLO11n()
device.deploy(model, data={
    "model_uri": "/models/yolo11n_imx_model/yolo11n_imx.onnx",
    "options": {"is_quantized": True},
})

annotator = Annotator()

with device as stream:
    for frame in stream:

        detections = frame.detections[frame.detections.confidence > 0.40]
        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)

        frame.display()
