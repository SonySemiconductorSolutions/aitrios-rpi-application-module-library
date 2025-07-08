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
from modlib.devices import AiCamera
from modlib.models.zoo import DeepLabV3Plus
from modlib.devices.frame import IMAGE_TYPE


class InstanceSegArgs:
    erosion_kernel: int = 3
    erosion_iteration: int = 2
    dilate_kernel: int = 5
    dilate_iteration: int = 5
    dist_threshold: float = 0.05
    size_threshold: int = 100
    config_mode = False


device = AiCamera()
model = DeepLabV3Plus()
device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:
        detections = frame.detections
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            detections.compensate_for_roi(frame.roi)
        instance_masks = detections.instance_segmentation(frame.width, frame.height, InstanceSegArgs)
        oriented_bboxes = detections.oriented_bbox()

        labels = [f"Class: {c}" for c, _, _, _ in detections]
        annotator.annotate_instance_segments(frame, detections)
        annotator.annotate_oriented_boxes(frame, detections, labels)
        frame.display()
