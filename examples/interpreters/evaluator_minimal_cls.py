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
import numpy as np
from datetime import datetime
from modlib.apps import Annotator
from modlib.devices import InterpreterClient, Frame, IMAGE_TYPE, Dataset
from modlib.models import COLOR_FORMAT, MODEL_TYPE, FRAMEWORK_FORMAT, Model
from modlib.models.evals import EvaluationSample, ClassificationEvaluator
from modlib.models.post_processors import pp_cls
from modlib.models.pre_processors import aspect_ratio_preserving_resize_with_pad, model_preprocess


# replace with proper models directory
MODELS="/path/to/models"


class YOLOClassification(Model):
    def __init__(self):
        # NOTE: This Sample Code is meant to be used with AI models such as Ultralytics YOLO. Please note that
        # the different license may apply to the AI model you would use and we may not be able to comply
        # your request for source codes except for the Sample Code. For example, Ultralytics YOLO is licensed by
        # Ultralytics Enterprise License, Ultralytics Academic License, or AGPL-3.0 License. If you want
        # to use the Ultralytics YOLO for commercial purpose, you need to purchase the Enterprise License from below.
        # [https://ultralytics.com/license]

        super().__init__(
            model_file=f"{MODELS}/yolo11n-cls_imx_model/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{MODELS}/yolo11n-cls_imx_model/labels.txt", dtype=str, delimiter="\n")
        self.norm_mean = 0.0
        self.norm_std = 255.0

    def _resize_fn(self, image):
        return aspect_ratio_preserving_resize_with_pad(image, self.input_tensor_size, pad_values=114)

    def pre_process(self, image, src_color_format, resize_fn=None):
        return model_preprocess(
            x=image,
            resize_fn=resize_fn if resize_fn is not None else self._resize_fn,
            src_color_format=src_color_format,
            model_color_format=self.color_format,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            framework_format=FRAMEWORK_FORMAT.CHW,
        )

    def post_process(self, output_tensors):
        return pp_cls(output_tensors)


model = YOLOClassification()
device = InterpreterClient(endpoint="http://localhost:8000")
device.deploy(model, data={
    # Path in mounted models directory of inference server
    "model_uri": "/models/yolo11n-cls_imx_model/yolo11n-cls_imx.onnx",
    "options": {"is_quantized": True},
})

dataset = Dataset(
    images_dir="examples/assets/imagenet_dataset",
    dataset_id_function=lambda path: int(path.parent.name)
)

all_samples: list[EvaluationSample] = []
annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)


with device:
    for sample in dataset:
        # 1. Prepare input tensor (prepare according to model preprocessing requirements)
        it_image, it, roi = model.pre_process(
            image=sample.image,
            src_color_format=sample.image_color_format,
            resize_fn=None,  # Use model's own resize function
        )

        # 2. Run model inference
        detections = device.infer(it)

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
