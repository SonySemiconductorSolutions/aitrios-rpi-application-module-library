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
from typing import Callable

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from modlib.apps.annotate import Annotator, Color
from modlib.models import COLOR_FORMAT, Detections, Poses
from modlib.models.evals import Evaluator, EvaluationSample
from modlib.devices.frame import Frame, IMAGE_TYPE


class COCOEvaluator(Evaluator):
    """
    Evaluator for COCO-format object detection.

    Converts detections to COCO JSON, runs `COCOeval` for bbox metrics, and
    can render ground truth vs detections overlays.
    """

    ground_truth: Path  #: Path to a COCO annotations file.
    label_mapping_func: Callable  #: Class-ID mapping function.

    def __init__(self, ground_truth: Path, num_classes: int = 80, label_mapping_func: Callable = None):
        """
        Initialize the COCO evaluator.

        Args:
            ground_truth: Path to a COCO annotations file.
            num_classes: Number of model classes. Used to pick the default mapping between 80- and 91-class IDs. Defaults to 80.
            label_mapping_func: Optional class-ID mapping function. If None, a default mapper is selected from `num_classes`.
        """
        self.ground_truth = ground_truth
        self.coco_gt = COCO(self.ground_truth)

        if label_mapping_func is not None:
            self.label_mapping_func = label_mapping_func
        else:
            # Default to mapping from 80 to 91 classes
            self.label_mapping_func = self._get_label_mapping_func(num_classes)

    def _get_label_mapping_func(self, num_classes: int) -> Callable:
        if num_classes == 80:
            return coco80_to_coco91
        elif num_classes == 91:
            return coco91_labels_to_coco_ids
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")

    def evaluate(self, samples: list[EvaluationSample]) -> np.ndarray:
        """
        Compute COCO bbox metrics over a list of evaluation samples.

        Args:
            samples: List of evaluation samples containing detections, ROI, and the dataset sample.

        Returns:
            Array containing COCOeval summary statistics.
        """
        img_ids = []
        coco_detections = []
        for s in samples:
            detections = s.detections
            image_path = s.dataset_sample.image_path
            image_id = s.dataset_sample.image_id
            image = s.dataset_sample.image

            # Scale detections to image
            scaled_detections = detections.copy()
            scaled_detections.compensate_for_roi(s.roi)
            if len(scaled_detections) > 0:
                scaled_detections.bbox[:, 0] = scaled_detections.bbox[:, 0] * image.shape[1]
                scaled_detections.bbox[:, 1] = scaled_detections.bbox[:, 1] * image.shape[0]
                scaled_detections.bbox[:, 2] = scaled_detections.bbox[:, 2] * image.shape[1]
                scaled_detections.bbox[:, 3] = scaled_detections.bbox[:, 3] * image.shape[0]

            scaled_detections.class_id = self.label_mapping_func(scaled_detections.class_id)

            # TO coco
            img_ids.append(image_id)
            img_info = self.coco_gt.loadImgs([image_id])[0]
            coco_height, coco_width = img_info["height"], img_info["width"]
            actual_height, actual_width = image.shape[0], image.shape[1]
            scale_x = coco_width / actual_width
            scale_y = coco_height / actual_height

            for bbox, confidence, class_id, _ in scaled_detections:
                coco_detections.append(
                    {
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            (bbox[2] - bbox[0]) * scale_x,
                            (bbox[3] - bbox[1]) * scale_y,
                        ],
                        "score": confidence,
                        "file": str(image_path),
                    }
                )

        # evaluation
        coco_dt = self.coco_gt.loadRes(coco_detections)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        if len(img_ids) > 0:
            coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        print(f"COCO evaluation results: {coco_eval.stats}")
        return coco_eval.stats

    def _get_detection_from_coco(self, image_id: int) -> Detections:
        """
        Load COCO ground-truth detections for specific image_id.
        """
        # Ground truth detections
        img_info = self.coco_gt.loadImgs([image_id])[0]
        coco_height, coco_width = img_info["height"], img_info["width"]
        ann_ids = self.coco_gt.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco_gt.loadAnns(ann_ids)

        coco_detections = Detections(
            bbox=np.array(
                [
                    [
                        ann["bbox"][0] / coco_width,
                        ann["bbox"][1] / coco_height,
                        (ann["bbox"][0] + ann["bbox"][2]) / coco_width,
                        (ann["bbox"][1] + ann["bbox"][3]) / coco_height,
                    ]
                    for ann in anns
                ]
            ),
            confidence=np.ones(len(anns), dtype=np.int32),  # ground truth confidence is 1.0
            class_id=np.array([ann["category_id"] for ann in anns]),
        )

        return coco_detections

    def visualize(
        self,
        samples: list[EvaluationSample],
        output_dir: Path,
        detection_threshold: float = 0.55,
    ) -> None:
        """
        Save images showing COCO ground truth (green) against detections (red).

        Args:
            samples: Evaluation samples with images, detections, and ROI info.
            output_dir: Directory where annotated images are written.
            detection_threshold: Confidence threshold applied to detections before drawing. Defaults to 0.55.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)

        for s in samples:
            detections = s.detections
            image_id = s.dataset_sample.image_id
            image = s.dataset_sample.image
            if s.dataset_sample.image_color_format == COLOR_FORMAT.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h, w, c = image.shape

            # Detections
            detections = detections[detections.confidence > detection_threshold]
            detections.compensate_for_roi(s.roi)
            detections.class_id = self.label_mapping_func(detections.class_id)

            # Ground truth detections
            coco_detections = self._get_detection_from_coco(image_id)

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

            annotator.annotate_boxes(frame, coco_detections, color=Color.green())
            annotator.annotate_boxes(frame, detections, color=Color.red())

            # Draw text labels for GT and DT
            cv2.putText(
                frame.image,
                "GT (Ground Truth)",
                (w - 155, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2,
                cv2.LINE_AA,
            )  # fmt: skip

            cv2.putText(
                frame.image,
                "DT (Detection)",
                (w - 125, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2,
                cv2.LINE_AA,
            )  # fmt: skip

            if s.roi != (0, 0, 1, 1):
                left, top, width, height = s.roi
                cv2.rectangle(
                    frame.image,
                    (int(left * w), int(top * h)),
                    (int((left + width) * w), int((top + height) * h)),
                    (0, 0, 255), 1
                )  # fmt: skip

            cv2.imwrite(output_dir / f"{image_id}_gt_vs_dt.jpg", frame.image)


def coco80_to_coco91(x: np.ndarray) -> np.ndarray:
    """
    COCO tools have 91 classes, but many coco trained models only return 80. This is because
    several classes have no images in the coco image dataset.

    Maps 80-class indices (0-79) to COCO 91-class IDs.
    """
    coco91Indexs = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
         63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90])  # fmt: skip
    return coco91Indexs[x.astype(np.int32)]


def coco91_labels_to_coco_ids(x: np.ndarray) -> np.ndarray:
    """
    Maps array indices from coco_labels_91.txt to COCO 91-class IDs.

    When a model uses coco_labels_91.txt, the model outputs are indices into that array.
    The array index corresponds to (COCO ID - 1) for most cases, but there are gaps
    (e.g., COCO ID 12 doesn't exist, so index 12 = COCO ID 13).

    This function maps: array_index -> COCO ID
    Since the labels file is 1-indexed (line N = COCO ID N) and arrays are 0-indexed,
    we simply add 1: COCO ID = array_index + 1
    """
    return (x + 1).astype(np.int32)


class COCOPoseEvaluator(Evaluator):
    """
    Evaluator for COCO-format human pose estimation.

    Converts keypoints to COCO format, runs `COCOeval` for keypoints, and can
    render ground truth vs predicted poses.
    """

    ground_truth: Path  #: Path to a COCO keypoints annotations file.

    def __init__(self, ground_truth: Path):
        """
        Initialize the COCO pose evaluator.

        Args:
            ground_truth: Path to a COCO keypoints annotations file.
        """
        self.ground_truth = ground_truth
        self.coco_gt = COCO(self.ground_truth)

    def evaluate(self, samples: list[EvaluationSample]):
        """
        Run COCO keypoint evaluation on a list of evaluation samples.

        Args:
            samples: Evaluation samples containing predicted poses and dataset metadata.

        Returns:
            Array containing COCOeval summary statistics for keypoints.
        """
        img_ids = []
        coco_detections = []
        for s in samples:
            detections = s.detections
            image_id = s.dataset_sample.image_id
            image = s.dataset_sample.image

            # Scale poses to image
            scaled_poses = detections.copy()
            scaled_poses.compensate_for_roi(s.roi)
            if scaled_poses.n_detections != 0 or scaled_poses.keypoints.size != 0:
                scaled_poses.keypoints[:, :, 0] = scaled_poses.keypoints[:, :, 0] * image.shape[1]
                scaled_poses.keypoints[:, :, 1] = scaled_poses.keypoints[:, :, 1] * image.shape[0]

            # TO coco
            img_ids.append(image_id)
            img_info = self.coco_gt.loadImgs([image_id])[0]
            coco_height, coco_width = img_info["height"], img_info["width"]
            actual_height, actual_width = image.shape[0], image.shape[1]
            scale_x = coco_width / actual_width
            scale_y = coco_height / actual_height

            for kp, confidence, keypoint_score, _, _ in scaled_poses:
                keypoints_coco = np.empty((len(kp), 3))  # fmt: skip
                keypoints_coco[:, 0] = np.clip(kp[:, 0] * scale_x, 0, coco_width - 1)
                keypoints_coco[:, 1] = np.clip(kp[:, 1] * scale_y, 0, coco_height - 1)
                keypoints_coco[:, 2] = np.where(
                    keypoint_score > 0.5, 2, 0
                )  # 2: labeled visible, 1: labeled not visible, 0: not visible
                num_visible = np.sum(keypoints_coco[:, 2] == 2)  # count visible keypoints

                # Skip when no visible keypoints to avoid COCOEval to count them as a detection
                if num_visible == 0:
                    continue

                coco_detections.append(
                    {
                        "image_id": int(image_id),
                        "category_id": 1,  # Person category in COCO
                        "keypoints": keypoints_coco.ravel(),
                        "num_keypoints": num_visible,
                        "score": confidence,
                    }
                )

        # Evaluation
        coco_dt = self.coco_gt.loadRes(coco_detections)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "keypoints")
        if len(img_ids) > 0:
            coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        print(f"COCO evaluation results: {coco_eval.stats}")
        return coco_eval.stats

    def _get_detection_from_coco(self, image_id: int) -> Poses:
        """
        Load COCO ground-truth poses for specific image_id.
        """
        # Ground truth detections
        img_info = self.coco_gt.loadImgs([image_id])[0]
        coco_height, coco_width = img_info["height"], img_info["width"]
        ann_ids = self.coco_gt.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco_gt.loadAnns(ann_ids)

        num_keypoints = 17
        keypoints = np.empty((0, num_keypoints, 2))
        keypoint_scores = np.empty((0, num_keypoints))
        for ann in anns:
            keypoints_coco = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            keypoint_scores_i = np.where(keypoints_coco[:, 2] == 2, 1.0, 0.0)
            keypoints_i = keypoints_coco[:, :2]
            keypoints_i[:, 0] /= coco_width
            keypoints_i[:, 1] /= coco_height
            keypoints = np.concatenate([keypoints, np.expand_dims(keypoints_i, axis=0)], axis=0)
            keypoint_scores = np.concatenate([keypoint_scores, np.expand_dims(keypoint_scores_i, axis=0)], axis=0)

        coco_detections = Poses(
            n_detections=len(anns),
            confidence=np.ones(len(anns), dtype=np.int32),  # ground truth confidence is 1.0
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
        )

        return coco_detections

    def visualize(
        self,
        samples: list[EvaluationSample],
        output_dir: Path,
        detection_threshold: float = 0.15,
        keypoint_score_threshold: float = 0.10,
    ) -> None:
        """
        Save images showing COCO ground-truth poses (green) against detections (red).

        Args:
            samples: Evaluation samples with images, predicted poses, and ROI info.
            output_dir: Directory where annotated images are written.
            detection_threshold: Confidence threshold applied to pose detections before drawing.
            keypoint_score_threshold: Minimum keypoint score to draw a predicted keypoint.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        annotator = Annotator(thickness=2, text_thickness=1, text_scale=0.4)

        for s in samples:
            detections = s.detections
            image_id = s.dataset_sample.image_id
            image = s.dataset_sample.image
            if s.dataset_sample.image_color_format == COLOR_FORMAT.RGB:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h, w, c = image.shape

            # Detections
            detections.compensate_for_roi(s.roi)
            detections = detections[detections.confidence > detection_threshold]

            # Ground truth detections
            coco_detections = self._get_detection_from_coco(image_id)

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

            annotator.annotate_keypoints(frame, coco_detections, keypoint_color=Color.green(), line_color=Color.green())
            annotator.annotate_keypoints(
                frame,
                detections,
                keypoint_color=Color.red(),
                line_color=Color.red(),
                keypoint_score_threshold=keypoint_score_threshold,
            )

            # Draw text labels for GT and DT
            cv2.putText(
                frame.image,
                "GT (Ground Truth)",
                (w - 155, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2,
                cv2.LINE_AA,
            )  # fmt: skip

            cv2.putText(
                frame.image,
                "DT (Detection)",
                (w - 125, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2,
                cv2.LINE_AA,
            )  # fmt: skip

            if s.roi != (0, 0, 1, 1):
                left, top, width, height = s.roi
                cv2.rectangle(
                    frame.image,
                    (int(left * w), int(top * h)),
                    (int((left + width) * w), int((top + height) * h)),
                    (0, 0, 255), 1
                )  # fmt: skip

            cv2.imwrite(output_dir / f"{image_id}_gt_vs_dt.jpg", frame.image)
