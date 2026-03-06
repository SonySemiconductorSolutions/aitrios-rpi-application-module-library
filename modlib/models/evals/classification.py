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

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from modlib.models import COLOR_FORMAT
from modlib.models.evals import Evaluator, EvaluationSample


class ClassificationEvaluator(Evaluator):
    """
    Evaluator for image classification outputs.
    Supports computing top-1 and top-5 accuracy and a confusion matrix.
    Uses ImageNet-v2 style ground truth, where each sample's image ID
    encodes its class.
    """

    ground_truth: Path  #: Directory containing ImageNet-v2 style class folders.
    labels: Optional[list[str]]  #: List of class names.
    num_classes: int  #: Number of classes. Defaults to 1000 ImageNet classes if not provided.

    def __init__(self, ground_truth: Path, labels: Optional[list[str]] = None):
        """
        Initialize the classification evaluator.

        Args:
            ground_truth: Directory containing ImageNet-v2 style class folders.
            labels: Optional list of class names; defaults to 1000 ImageNet classes when None.

        Raises:
            FileNotFoundError: If the provided ground_truth path is not a directory.
        """
        self.ground_truth = Path(ground_truth)
        if not self.ground_truth.is_dir():
            raise FileNotFoundError(
                f"Ground truth must be a directory for ImageNet-v2 folder-based structure. Got: {self.ground_truth}"
            )

        self.labels = labels
        self.num_classes = len(labels) if labels is not None else 1000  # Default to 1000 imagenet classes

    def _get_ground_truth_class(self, image_id: Union[int, str]) -> Optional[int]:
        """
        Get ground truth class ID for an image.
        For ImageNet-v2 folder-based structure, image_id is the class ID itself
        (extracted from the folder name).
        """
        if isinstance(image_id, int):
            # Validate it's in valid range
            if 0 <= image_id < self.num_classes:
                return image_id
        return None

    def _print_results(
        self,
        total: int,
        correct_top1: int,
        correct_top5: int,
        top1_accuracy: float,
        top5_accuracy: float,
        confusion_matrix: np.ndarray,
        per_class_accuracy: np.ndarray,
        per_class_correct: np.ndarray,
        per_class_total: np.ndarray,
    ):
        print("\n" + "=" * 80)
        print("Classification Evaluation Results")
        print("=" * 80)
        print(f"Total samples: {total}")
        print(f"Top-1 Accuracy: {top1_accuracy:.4f} ({correct_top1}/{total})")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f} ({correct_top5}/{total})")

        # Print per-class accuracy for classes that have samples
        classes_with_samples = [i for i in range(self.num_classes) if per_class_total[i] > 0]
        if classes_with_samples:
            print(f"\nPer-class accuracy ({len(classes_with_samples)} classes with samples):")
            for i in classes_with_samples:
                class_name = self.labels[i] if self.labels is not None and i < len(self.labels) else f"Class_{i}"
                print(f"  {class_name}: {per_class_accuracy[i]:.4f} ({per_class_correct[i]}/{per_class_total[i]})")

        # Print confusion matrix for classes that have samples
        if classes_with_samples:
            print("\nConfusion Matrix (rows=ground truth, cols=predicted):")
            # Print header with class names/indices
            max_class_idx = max(classes_with_samples)
            header = "GT\\Pred"
            for j in range(min(max_class_idx + 1, self.num_classes)):
                if per_class_total[j] > 0 or any(confusion_matrix[i, j] > 0 for i in classes_with_samples):
                    header += f"  {j:>4}"
            print(header)
            # Print matrix rows for classes with samples
            for i in classes_with_samples:
                class_name = self.labels[i] if self.labels is not None and i < len(self.labels) else f"Class_{i}"
                row_str = f"{i:>3} ({class_name[:15]:<15})"
                for j in range(min(max_class_idx + 1, self.num_classes)):
                    if per_class_total[j] > 0 or any(confusion_matrix[k, j] > 0 for k in classes_with_samples):
                        row_str += f"  {confusion_matrix[i, j]:>4}"
                print(row_str)

        print("=" * 80)

    def evaluate(self, samples: list[EvaluationSample]):
        """
        Compute top-1/top-5 accuracy and confusion matrix for classification predictions.

        Args:
            samples: Samples containing dataset images, ground-truth IDs, and predicted classifications.

        Returns:
            Dict with aggregate metrics, per-class metrics, and the confusion matrix.
        """
        total = 0
        correct_top1 = 0
        correct_top5 = 0
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        for s in samples:
            gt_class = self._get_ground_truth_class(s.dataset_sample.image_id)
            if gt_class is None:
                print(f"Warning: No ground truth for image_id {s.dataset_sample.image_id}, skipping")
                continue

            if len(s.detections.class_id) == 0:
                print(f"Warning: No predictions for image_id {gt_class}, skipping")
                continue

            # Sort by confidence to get top predictions
            # y_pred_k = k-th highest confidence prediction
            sorted_indices = np.argsort(s.detections.confidence)[::-1]
            class_id_array = np.asarray(s.detections.class_id)
            top1_class = class_id_array[sorted_indices[0]]  # y_pred_1
            top5_classes = class_id_array[sorted_indices[:5]]  # {y_pred_1, ..., y_pred_5}

            # Update metrics according to formulas in module docstring:
            # Top1Acc = (1/N) * sum_i [y_pred_1[i] == y_true[i]]
            # Top5Acc = (1/N) * sum_i [y_true[i] in {y_pred_1[i], ..., y_pred_5[i]}]
            total += 1
            if top1_class == gt_class:
                correct_top1 += 1  # Count: y_pred_1 == y_true
            if gt_class in top5_classes:
                correct_top5 += 1  # Count: y_true in {y_pred_1, ..., y_pred_5}

            # Update confusion matrix: CM[c_gt, c_pred] = count of samples with GT=c_gt, pred=c_pred
            # Diagonal elements (CM[c, c]) = TP_c (true positives for class c)
            if 0 <= gt_class < self.num_classes and 0 <= top1_class < self.num_classes:
                confusion_matrix[gt_class, top1_class] += 1

        # Compute final metrics
        # Top-1 Accuracy: Top1Acc = correct_top1 / total = (1/N) * sum_i [y_pred_1[i] == y_true[i]]
        top1_accuracy = correct_top1 / total if total > 0 else 0.0

        # Top-5 Accuracy: Top5Acc = correct_top5 / total = (1/N) * sum_i [y_true[i] in {y_pred_1[i], ..., y_pred_5[i]}]
        top5_accuracy = correct_top5 / total if total > 0 else 0.0

        # Per-class accuracy: ClassAcc_c = TP_c / (TP_c + FN_c)
        # TP_c = diagonal element CM[c, c] (correct predictions for class c)
        # (TP_c + FN_c) = row sum CM[c, :] (total samples with GT = c)
        per_class_correct = np.diag(confusion_matrix)  # TP_c for each class
        per_class_total = confusion_matrix.sum(axis=1)  # (TP_c + FN_c) for each class
        # Use np.divide with where to avoid division by zero warning
        per_class_accuracy = np.divide(
            per_class_correct,
            per_class_total,
            out=np.zeros_like(per_class_correct, dtype=float),
            where=per_class_total > 0,
        )

        # Print results
        self._print_results(
            total=total,
            correct_top1=correct_top1,
            correct_top5=correct_top5,
            top1_accuracy=top1_accuracy,
            top5_accuracy=top5_accuracy,
            confusion_matrix=confusion_matrix,
            per_class_accuracy=per_class_accuracy,
            per_class_correct=per_class_correct,
            per_class_total=per_class_total,
        )

        return {
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "total_samples": total,
            "correct_top1": correct_top1,
            "correct_top5": correct_top5,
            "confusion_matrix": confusion_matrix,
            "per_class_accuracy": per_class_accuracy,
        }

    def visualize(self, samples: list[EvaluationSample], output_dir: Path) -> None:
        """
        Save visualizations overlaying ground-truth labels and top predictions on images.

        Args:
            samples: Samples to visualize.
            output_dir: Directory where annotated images are written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for s in samples:
            image = s.dataset_sample.image
            if s.dataset_sample.image_color_format == COLOR_FORMAT.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gt_class = self._get_ground_truth_class(s.dataset_sample.image_id)
            if gt_class is None:
                print(f"Warning: No ground truth for image_id {s.dataset_sample.image_id}, skipping")
                continue

            # Draw ground truth label
            text = f"GT: {self.labels[gt_class]}({gt_class})"
            cv2.putText(image, text, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw detections labels
            for i, label_id in enumerate(s.detections.class_id[:3]):
                text = f"{i + 1}. {self.labels[label_id]}({label_id}): {s.detections.confidence[i]:.2f}"
                cv2.putText(image, text, (50, 30 + 25 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if s.roi != (0, 0, 1, 1):
                h, w, _ = image.shape
                left, top, width, height = s.roi
                cv2.rectangle(
                    image,
                    (int(left * w), int(top * h)),
                    (int((left + width) * w), int((top + height) * h)),
                    (0, 0, 255), 1
                )  # fmt: skip

            # Save image
            cv2.imwrite(output_dir / f"{s.dataset_sample.image_id}_gt_vs_dt.jpg", image)
