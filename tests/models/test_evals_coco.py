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

import os
import pytest
import numpy as np
from pathlib import Path

from modlib.models.results import ROI
from modlib.devices import Dataset
from modlib.models.evals import COCOEvaluator, COCOPoseEvaluator, EvaluationSample
from tests.utils import get_coco_annotations, get_coco_samples, get_coco_keypoints_samples


@pytest.fixture()
def coco_annotations():
    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/coco_annotations"
    _ = get_coco_annotations(current_dir)
    return Path(current_dir)


@pytest.fixture()
def coco_samples():
    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/coco_samples"
    _ = get_coco_samples(current_dir)
    return Path(current_dir)


@pytest.fixture()
def coco_keypoints_samples():
    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/coco_keypoints_samples"
    _ = get_coco_keypoints_samples(current_dir)
    return Path(current_dir)


#################################
### COCO Detection Evaluation ###
#################################

def test_gt_non_exist():
    with pytest.raises(FileNotFoundError):
        COCOEvaluator(Path("path/to/non_exist"))


def test_with_ground_truth(coco_annotations, coco_samples):
    evaluator = COCOEvaluator(
        coco_annotations / "instances_val2017.json",
        label_mapping_func=lambda x: x # identity mapping since ground truth already has COCO IDs
    )
    dataset = Dataset(coco_samples)
    samples = []
    for s in dataset:
        samples.append(EvaluationSample(
            roi=ROI(0, 0, 1, 1),
            detections=evaluator._get_detection_from_coco(s.image_id),
            dataset_sample=s
        ))

    r = evaluator.evaluate(samples)
    r = np.delete(r, 6)  # NOTE: Average Recall @ maxDets = 1, but there are more 100% confidence detections
    assert np.all(r == 1.0)


#################################
### COCO Keypoints Evaluation ###
#################################

def test_gt_non_exist_keypoints():
    with pytest.raises(FileNotFoundError):
        COCOPoseEvaluator(Path("path/to/non_exist"))


def test_with_ground_truth_keypoints(coco_annotations, coco_keypoints_samples):
    evaluator = COCOPoseEvaluator(coco_annotations / "person_keypoints_val2017.json")
    dataset = Dataset(coco_keypoints_samples)
    samples = []
    for s in dataset:
        samples.append(EvaluationSample(
            roi=ROI(0, 0, 1, 1),
            detections=evaluator._get_detection_from_coco(s.image_id),
            dataset_sample=s
        ))

    r = evaluator.evaluate(samples)
    assert np.all(r == 1.0)
