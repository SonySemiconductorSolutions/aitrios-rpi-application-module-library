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

from modlib.models.results import ROI, Classifications
from modlib.devices import Dataset
from modlib.models.evals import ClassificationEvaluator, EvaluationSample
from tests.utils import get_imagenet_dataset


@pytest.fixture
def imagenet_dataset():

    current_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../assets/imagenet_dataset"
    _ = get_imagenet_dataset(current_dir)

    return Path(current_dir)


def test_gt_non_exist():
    with pytest.raises(FileNotFoundError):
        ClassificationEvaluator(Path("path/to/non_exist"))


def test_with_ground_truth(imagenet_dataset):

    dataset = Dataset(imagenet_dataset, dataset_id_function=lambda path: int(path.parent.name))
    evaluator = ClassificationEvaluator(imagenet_dataset)

    samples = []
    for sample in dataset:
        samples.append(EvaluationSample(
            roi=ROI(0, 0, 1, 1),
            detections=Classifications(class_id=np.array([sample.image_id]), confidence=np.array([1.0])),
            dataset_sample=sample
        ))

    r = evaluator.evaluate(samples)
    sample_ids = [sample.dataset_sample.image_id for sample in samples]

    # Perfect match
    assert np.all(r["per_class_accuracy"][sample_ids] == 1) # 100%
    assert np.all(r["confusion_matrix"][sample_ids, sample_ids] == 1) # 1 sample per class
