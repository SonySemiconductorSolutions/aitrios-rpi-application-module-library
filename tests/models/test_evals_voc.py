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
from pathlib import Path

from modlib.models.results import ROI, Segments
from modlib.devices import Dataset
from modlib.models.evals import VocSegEvaluator, EvaluationSample
from tests.utils import get_voc_samples


@pytest.fixture()
def voc_samples():

    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/voc_samples"
    _ = get_voc_samples(current_dir)

    return Path(current_dir)


def test_gt_non_exist():
    with pytest.raises(FileNotFoundError):
        VocSegEvaluator(Path("path/to/non_exist"))


def test_with_ground_truth(voc_samples):
    evaluator = VocSegEvaluator(voc_samples / "SegmentationClass")
    dataset = Dataset(voc_samples / "JPEGImages")
    samples = []
    for sample in dataset:
        samples.append(EvaluationSample(
            roi=ROI(0, 0, 1, 1),
            detections=Segments(mask=evaluator._load_gt(sample.image_id)),
            dataset_sample=sample
        ))
    r = evaluator.evaluate(samples)

    assert r['mean_iou'] == 1
    assert r['mean_accuracy'] == 1
    assert r['pixel_accuracy'] == 1
    assert r['fw_iou'] == 1
