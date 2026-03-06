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
from modlib.devices import AiCamera, Frame, IMAGE_TYPE, Dataset
from modlib.models.zoo import HigherHRNet
from modlib.models.evals import EvaluationSample, COCOPoseEvaluator
from modlib.devices.imx500 import isp


device = AiCamera(data_injection=True, frame_rate=20)
model = HigherHRNet()
device.deploy(model)

dataset = Dataset(
    images_dir="examples/assets/coco_keypoints_samples",
    dataset_id_function=None,  # Use default coco dataset_id_function
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

        detections = frame.detections[frame.detections.confidence > 0.15]
        annotator.annotate_keypoints(frame, detections, keypoint_score_threshold=0.10)

        frame.display()

# Evaluate
evaluator = COCOPoseEvaluator(ground_truth="examples/assets/coco_annotations/person_keypoints_val2017.json")

evaluator.evaluate(all_samples)
evaluator.visualize(
    samples=all_samples,
    output_dir="tmp/annotated_images_pose",
    detection_threshold=0.15,
    keypoint_score_threshold=0.10,
)
