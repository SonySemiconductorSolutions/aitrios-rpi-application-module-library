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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from modlib.models.results import ROI, Detections, Segments, Classifications, Poses
from modlib.devices.sources import DatasetSample


@dataclass
class EvaluationSample:
    """
    Data class to store a sample ready for evaluation.
    """

    roi: ROI  #: Region of Interest (normalized - left, top, width, height) to compensate for.
    detections: Detections | Segments | Classifications | Poses  #: Result to evaluate.
    dataset_sample: DatasetSample  #: Dataset sample containing the image and ground truth.


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        samples: list[EvaluationSample],
        *args,
    ) -> None:
        """
        Abstract method to evaluate model predictions.
        Should be implemented by subclasses.
        """
        ...

    @abstractmethod
    def visualize(
        self,
        samples: list[EvaluationSample],
        output_dir: Path,
        *args,
    ) -> None:
        """
        Abstract method to visualize model predictions.
        Should be implemented by subclasses.
        """
        ...
