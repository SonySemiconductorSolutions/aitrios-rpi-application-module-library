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

import cv2
from datetime import datetime
from modlib.apps import Annotator
from modlib.devices import AiCamera, Frame, IMAGE_TYPE, Dataset
from modlib.models.zoo import MobileNetV2
from modlib.models.evals import EvaluationSample, ClassificationEvaluator
from modlib.devices.imx500 import isp


device = AiCamera(data_injection=True, frame_rate=20)
model = MobileNetV2()
device.deploy(model)

dataset = Dataset(
    images_dir="examples/assets/imagenet_dataset",
    dataset_id_function=lambda path: int(path.parent.name)
)

all_samples: list[EvaluationSample] = []
annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)


with device:
    for sample in dataset:
        # 1. Prepare input tensor (like IMX500 ISP)
        dsp_input_tensor, roi = isp.prepare_tensor_like_isp(
            img=sample.image,
            model=model,
            src_color_format=sample.image_color_format,
        )

        # 2. Inject input tensor
        detections = device.inject(dsp_input_tensor)

        # 3. Store Evaluation Sample
        all_samples.append(EvaluationSample(
            roi=roi,
            detections=detections,
            dataset_sample=sample,
        ))

        # Vizualize result on sample image
        h, w, c = sample.image.shape
        frame = Frame(
            timestamp=datetime.now(),
            image=sample.image.copy(), image_type=IMAGE_TYPE.SOURCE,
            color_format=sample.image_color_format,
            width=w, height=h, channels=c,
            detections=detections, roi=roi,
            new_detection=True, fps=0, dps=0,
            frame_count=0,
        )

        for i, label in enumerate([model.labels[id] for id in detections.class_id[:3]]):
            text = f"{i + 1}. {label}: {detections.confidence[i]:.2f}"
            cv2.putText(frame.image, text, (50, 30 + 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)

        frame.display()

# Evaluate
evaluator = ClassificationEvaluator(
    ground_truth="examples/assets/imagenet_dataset",
    labels=model.labels,
)

evaluator.evaluate(all_samples)
evaluator.visualize(all_samples, output_dir="tmp/annotated_images_cls")
