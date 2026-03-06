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
from typing import Union, Optional, Dict, Callable

import cv2
import copy
import numpy as np
from PIL import Image

from modlib.apps.annotate import Annotator
from modlib.models import Segments
from modlib.models.evals import Evaluator, EvaluationSample
from modlib.devices.frame import Frame, IMAGE_TYPE, COLOR_FORMAT


class VocSegEvaluator(Evaluator):
    """
    Evaluator for semantic segmentation on VOC-style datasets.

    Computes confusion matrix-derived metrics (mIoU, pixel accuracy, mean
    accuracy, frequency weighted IoU) and can visualize predictions against
    ground-truth masks.
    """

    ground_truth: Path  #: Directory containing per-image PNG masks named `<image_id>.png`.
    num_classes: int  #: Number of semantic classes. Defaults to 21.
    ignore_index: int  #: Label value to ignore when computing metrics. Defaults to 255.
    label_mapping_func: Callable  #: Class-ID mapping function.

    def __init__(
        self,
        ground_truth: Path,
        num_classes: int = 21,
        ignore_index: int = 255,
        label_mapping_func: Callable = None,
    ):
        """
        Initialize a VOC segmentation evaluator.

        Args:
            ground_truth: Directory containing per-image PNG masks named `<image_id>.png`.
            num_classes: Number of semantic classes. Defaults to 21.
            ignore_index: Label value to ignore when computing metrics. Defaults to 255.
            label_mapping_func: Optional class-ID mapping function.
        """
        self.ground_truth = Path(ground_truth)
        if not self.ground_truth.is_dir():
            raise FileNotFoundError(
                f"Ground truth must be a directory for VOC segmentation masks. Got: {self.ground_truth}"
            )

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if label_mapping_func is None:
            # Modlib's segmentation masks use background pixels as -1/255, but VOC uses 0.
            self.label_mapping_func = lambda x: np.where(x == 255, 0, x)
        else:
            self.label_mapping_func = label_mapping_func

    def _load_gt(self, image_id: Union[int, str]) -> np.ndarray:
        """Load a ground-truth mask for an image."""
        gt_path = self.ground_truth / f"{image_id}.png"
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth mask not found: {gt_path}")

        with Image.open(gt_path) as img:
            return np.array(img, dtype=np.int64)

    def _to_class_mask(
        self,
        prediction: np.ndarray,
        target_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Convert prediction to class mask format.
        Accepted input:
        - [H, W]        -> already class IDs
        - [C, H, W]     -> logits/probabilities
        """
        if prediction.ndim == 2:
            mask = prediction
        elif prediction.ndim == 3:
            mask = prediction.argmax(axis=0)
        else:
            raise ValueError(f"Unsupported prediction shape: {prediction.shape}")

        # Resize prediction to match ground truth shape
        if mask.shape != target_shape:
            th, tw = target_shape
            img = Image.fromarray(mask.astype(np.int32))
            img = img.resize((tw, th), resample=Image.NEAREST)
            mask = np.array(img, dtype=np.int64)

        # Apply label mapping function
        if self.label_mapping_func is not None:
            mask = self.label_mapping_func(mask)

        return mask.astype(np.int64)

    def evaluate(self, samples: list[EvaluationSample]) -> Dict:
        """
        Compute segmentation metrics over a list of evaluation samples.

        Args:
            samples: Collection of evaluation samples with detections and ground-truth references.

        Returns:
            Dictionary containing confusion-matrix derived metrics.
        """
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        for s in samples:
            gt_mask = self._load_gt(s.dataset_sample.image_id)
            pred_mask = self._to_class_mask(s.detections.mask, gt_mask.shape)

            cm += _fast_confusion_matrix(
                pred_mask,
                gt_mask,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index,
            )

        metrics = _metrics_from_cm(cm)

        # Print summary
        print("\n" + "=" * 100)
        print(f"mIoU: {metrics['mean_iou']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
        print(f"Frequency Weighted IoU: {metrics['fw_iou']:.4f}")
        print("=" * 100)

        return metrics

    def visualize(self, samples: list[EvaluationSample], output_dir: Path) -> None:
        """
        Save side-by-side ground-truth and detection overlays for inspection.

        Args:
            samples: List of evaluation samples containing images, predictions, and ROI info.
            output_dir: Directory where comparison images are written.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)

        output_dir = Path(output_dir)
        for s in samples:
            detections = s.detections
            image_id = s.dataset_sample.image_id
            image = s.dataset_sample.image
            if s.dataset_sample.image_color_format == COLOR_FORMAT.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h, w, c = image.shape

            # Detections
            detections.compensate_for_roi(s.roi)

            frame = Frame(
                timestamp=None,
                image=image,
                image_type=IMAGE_TYPE.SOURCE,
                width=w, height=h, channels=c,
                detections=None,
                new_detection=True,
                fps=0, dps=0,
                color_format=COLOR_FORMAT.BGR,
                roi=(0, 0, 1, 1),
            )  # fmt: skip

            dt_ann = annotator.annotate_segments(copy.copy(frame), detections)

            if s.roi != (0, 0, 1, 1):
                left, top, width, height = s.roi
                cv2.rectangle(
                    dt_ann,
                    (int(left * w), int(top * h)),
                    (int((left + width) * w), int((top + height) * h)),
                    (0, 0, 255), 1
                )  # fmt: skip

            # Ground truth
            gt_mask = self._load_gt(image_id)
            gt_detections = Segments(mask=gt_mask)
            gt_detections.mask = np.where(gt_detections.mask == 0, 255, gt_detections.mask)
            gt_ann = annotator.annotate_segments(copy.copy(frame), gt_detections)
            combined_image = np.hstack([gt_ann, dt_ann])

            # Add labels
            cv2.putText(
                combined_image,
                "GT (Ground Truth)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1,
                cv2.LINE_AA,
            )  # fmt: skip

            cv2.putText(
                combined_image,
                "DT (Detection)",
                (w + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 255), 1,
                cv2.LINE_AA,
            )  # fmt: skip

            cv2.imwrite(output_dir / f"{image_id}_gt_vs_dt.jpg", combined_image)


def _fast_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """Build a confusion matrix from predicted and target masks."""
    assert pred.shape == target.shape

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]

    indices = target * num_classes + pred
    cm = np.bincount(indices, minlength=num_classes**2)
    cm = cm.reshape(num_classes, num_classes)
    return cm


def _metrics_from_cm(cm: np.ndarray) -> Dict:
    """Compute segmentation metrics from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    gt = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)

    denom = gt + pred - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom != 0)

    cls_acc = np.divide(tp, gt, out=np.zeros_like(tp), where=gt != 0)

    total = cm.sum()
    pixel_acc = float(tp.sum() / total) if total > 0 else 0.0

    valid = gt > 0
    mean_iou = float(iou[valid].mean()) if np.any(valid) else 0.0
    mean_acc = float(cls_acc[valid].mean()) if np.any(valid) else 0.0

    freq = gt / gt.sum() if gt.sum() > 0 else np.zeros_like(gt)
    fw_iou = float((freq[freq > 0] * iou[freq > 0]).sum()) if np.any(freq > 0) else 0.0

    return {
        "per_class_iou": iou,
        "mean_iou": mean_iou,
        "per_class_accuracy": cls_acc,
        "mean_accuracy": mean_acc,
        "pixel_accuracy": pixel_acc,
        "fw_iou": fw_iou,
    }


def coco80_to_voc21(x: np.ndarray) -> np.ndarray:
    """
    Maps COCO 80-class indices to VOC 21-class indices.
    """
    coco_to_voc = {
        # 255: 0,  # background
        0: 15,  # person
        1: 2,  # bicycle
        2: 7,  # car
        3: 14,  # motorcycle -> motorbike
        4: 1,  # airplane -> aeroplane
        5: 6,  # bus
        6: 19,  # train
        8: 4,  # boat
        14: 3,  # bird
        15: 8,  # cat
        16: 12,  # dog
        17: 13,  # horse
        18: 17,  # sheep
        19: 10,  # cow
        39: 5,  # bottle
        56: 9,  # chair
        57: 18,  # couch -> sofa
        58: 16,  # potted plant -> pottedplant
        60: 11,  # dining table -> diningtable
        62: 20,  # tv -> tvmonitor
    }

    lut = np.zeros(256, dtype=np.int64)
    for coco_id, voc_id in coco_to_voc.items():
        lut[coco_id] = voc_id

    return lut[x.astype(np.int32)]
