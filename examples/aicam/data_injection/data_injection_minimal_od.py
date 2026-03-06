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

from datetime import datetime
from modlib.apps import Annotator
from modlib.devices import AiCamera, Frame, IMAGE_TYPE
from modlib.models.zoo import NanoDetPlus416x416
from modlib.devices.sources import Images
from modlib.devices.imx500 import isp


device = AiCamera(data_injection=True, frame_rate=20)
model = NanoDetPlus416x416()
device.deploy(model)

source = Images("examples/assets/coco_samples")
annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)


with device:
    for img in source:
        # 1. Prepare input tensor (like IMX500 ISP)
        dsp_input_tensor, roi = isp.prepare_tensor_like_isp(
            img=img,
            model=model,
            src_color_format=source.color_format
        )

        # 2. Inject input tensor
        detections = device.inject(dsp_input_tensor)

        # 3. Visualize result with a Frame
        h, w, c = img.shape
        frame = Frame(
            timestamp=datetime.now(),
            image=img, image_type=IMAGE_TYPE.SOURCE,
            color_format=source.color_format,
            width=w, height=h, channels=c,
            detections=detections, roi=roi,
            new_detection=True, fps=0, dps=0,
            frame_count=0,
        )

        detections = frame.detections[frame.detections.confidence > 0.40]
        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)

        frame.display()
