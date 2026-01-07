#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
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

import base64
import gzip
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Iterator, List, Tuple, Union

import cv2
import numpy as np


@dataclass
class ROI:
    """
    Region of Interest (ROI) specifying the bounding box coordinates.
    """

    left: float  #: The x-coordinate of the ROI normalized to the width of the frame.
    top: float  #: The y-coordinate of the ROI normalized to the height of the frame.
    width: float  #: The width of the ROI normalized to the width of the frame.
    height: float  #: The height of the ROI normalized to the height of the frame.

    def json(self) -> dict:
        """
        Convert the ROI to a JSON-serializable dictionary.
        """
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "ROI":
        """
        Create an ROI from a JSON-serializable dictionary.
        """
        return cls(**{k: float(data[k]) for k in ("left", "top", "width", "height")})

    def __getitem__(self, index: int) -> float:
        return (self.left, self.top, self.width, self.height)[index]

    def __iter__(self):
        """
        Iterates over the ROI. (left, top, width, height)
        """
        return iter((self.left, self.top, self.width, self.height))


class Result(ABC):
    """
    Abstract base class for a model detection result type.
    """

    @abstractmethod
    def compensate_for_roi(self, roi: ROI):
        """
        Abstract method responsible for aligning the current detection type with
        the corresponding `frame.image`. One needs guarantee the resulting detections
        are compensated for any possible ROI that may be applied.

        Args:
            roi: The ROI (normalized - left, top, width, height) to compensate for.
        """
        pass

    @abstractmethod
    def json(self) -> dict:
        """
        Convert the result object to a JSON-serializable dictionary.
        """
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: dict):
        """
        Create a Result object from a JSON-serializable dictionary.
        Returns the result object in the corresponding result type.
        """
        pass


class Classifications(Result):
    """
    Data class for classification results.
    """

    confidence: np.ndarray  #: Array of shape (n,) representing the confidence of N detections.
    class_id: np.ndarray  #: Array of shape (n,) representing the class id of N detections.

    def __init__(
        self,
        confidence: np.ndarray = np.empty((0,)),
        class_id: np.ndarray = np.empty((0,)),
    ) -> None:
        """
        Initialize a new instance of Classifications.

        Args:
            confidence: Array of shape (n,) representing the confidence of N detections.
            class_id: Array of shape (n,) representing the class id of N detections.
        """
        self.confidence = confidence
        self.class_id = class_id

    def compensate_for_roi(self, roi: ROI):
        pass

    def __len__(self):
        """
        Returns the number of detections.
        """
        return len(self.class_id)

    def __copy__(self) -> "Classifications":
        """
        Returns a copy of the current detections.
        """
        new_instance = Classifications()
        new_instance.confidence = np.copy(self.confidence)
        new_instance.class_id = np.copy(self.class_id)

        return new_instance

    def copy(self) -> "Classifications":
        """
        Returns a copy of the current detections.
        """
        return self.__copy__()

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "Classifications":
        """
        Returns a new Classifications object with the selected detections.
        Could be a subsection of the current detections.

        Args:
            index: The index or indices of the detections to select.

        Returns:
            A new Classifications object with the selected detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.confidence = self.confidence[index]
        res.class_id = self.class_id[index]
        return res

    def __iter__(self) -> Iterator[Tuple[float, int]]:
        """
        Iterate over the detections.

        Yields:
            Tuple[float, int]: A tuple containing the confidence and class id of each detection.
        """
        for i in range(len(self)):
            yield (
                self.confidence[i],
                self.class_id[i],
            )

    def __add__(self, other: "Classifications") -> "Classifications":
        """
        Concatenate two Classifications objects.

        Args:
            other: The other Classifications object to concatenate.

        Returns:
            The concatenated Classifications.
        """
        if not isinstance(other, Classifications):
            raise TypeError(f"Unsupported operand type(s) for +: 'Classifications' and '{type(other)}'")

        result = self.copy()
        result.confidence = np.concatenate([self.confidence, other.confidence])
        result.class_id = np.concatenate([self.class_id, other.class_id])
        return result

    def __str__(self) -> str:
        """
        Return a string representation of the Classifications object.

        Returns:
            A string representation of the Classifications object.
        """
        return f"Classifications(class_id:\t {self.class_id}, \tconfidence:\t {self.confidence})"

    def json(self) -> dict:
        """
        Convert the Classifications object to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the Classifications object with the following keys:
            - "confidence" (list): The confidence scores.
            - "class_id" (list): The class IDs.
        """
        return {
            "confidence": self.confidence.tolist(),
            "class_id": self.class_id.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict) -> "Classifications":
        """
        Create a Classifications instance from a JSON-serializable dictionary.

        Args:
            data: JSON-serializable dictionary with classification data.

        Returns:
            The Classifications instance created from the JSON data.
        """
        confidence = np.array(data["confidence"], dtype=np.float32)
        class_id = np.array(data["class_id"], dtype=np.int32)

        return cls(confidence=confidence, class_id=class_id)


class Detections(Result):
    """
    Data class for object detections.
    """

    bbox: np.ndarray  #: Array of shape (n, 4) the bounding boxes [x1, y1, x2, y2] of N detections
    confidence: np.ndarray  #: Array of shape (n,) the confidence of N detections
    class_id: np.ndarray  #: Array of shape (n,) the class id of N detections
    tracker_id: np.ndarray  #: Array of shape (n,) the tracker id of N detections

    def __init__(
        self,
        bbox: np.ndarray = np.empty((0, 4)),
        confidence: np.ndarray = np.empty((0,)),
        class_id: np.ndarray = np.empty((0,)),
    ):
        """
        Initialize the Detections object.

        Args:
            bbox: Array of shape (n, 4) the bounding boxes [x1, y1, x2, y2] of N detections
            confidence: Array of shape (n,) the confidence of N detections
            class_id: Array of shape (n,) the class id of N detections
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = None

        self._roi_compensated = False

    def compensate_for_roi(self, roi: ROI):
        """
        Compensate the bounding boxes for the given ROI.

        Args:
            roi: The ROI (normalized - left, top, width, height) to compensate for.
        """
        if roi == (0, 0, 1, 1) or self._roi_compensated:
            return

        self.bbox[:, 0] = roi[0] + self.bbox[:, 0] * roi[2]
        self.bbox[:, 1] = roi[1] + self.bbox[:, 1] * roi[3]
        self.bbox[:, 2] = roi[0] + self.bbox[:, 2] * roi[2]
        self.bbox[:, 3] = roi[1] + self.bbox[:, 3] * roi[3]

        self._roi_compensated = True

    def __len__(self) -> int:
        """
        Returns the number of detections.
        """
        return len(self.class_id)

    def __copy__(self) -> "Detections":
        """
        Returns a copy of the current detections.
        """
        new_instance = Detections()
        new_instance.bbox = np.copy(self.bbox)
        new_instance.confidence = np.copy(self.confidence)
        new_instance.class_id = np.copy(self.class_id)
        new_instance.tracker_id = np.copy(self.tracker_id)
        new_instance._roi_compensated = copy.deepcopy(self._roi_compensated)
        return new_instance

    def copy(self) -> "Detections":
        """
        Returns a copy of the current detections.
        """
        return self.__copy__()

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "Detections":
        """
        Returns a new Detections object with the selected detections.

        Args:
            index: The index or indices of the detections to select.

        Returns:
            A new Detections object with the selected detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.confidence = self.confidence[index]
        res.class_id = self.class_id[index]
        res.bbox = self.bbox[index] if self.bbox is not None else None
        res.tracker_id = self.tracker_id[index] if self.tracker_id is not None else None
        return res

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, int]]:
        """
        To iterate over the detections.
        """
        for i in range(len(self)):
            yield (
                self.bbox[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    def __add__(self, other: "Detections") -> "Detections":
        """
        Concatenate two Detections objects.

        Args:
            other: The other Detections object to concatenate.

        Returns:
            The concatenated Detections.
        """
        if not isinstance(other, Detections):
            raise TypeError(f"Unsupported operand type(s) for +: 'Detections' and '{type(other)}'")

        result = self.copy()
        result.bbox = np.vstack((result.bbox, other.bbox))
        result.confidence = np.concatenate([self.confidence, other.confidence])
        result.class_id = np.concatenate([self.class_id, other.class_id])
        result.tracker_id = np.concatenate([self.tracker_id, other.tracker_id]) if self.tracker_id is not None else None
        return result

    def __str__(self) -> str:
        """
        Return a string representation of the Detections object.

        Returns:
            A string representation of the Detections object.
        """
        s = f"Detections(class_id:\t {self.class_id}, \tconfidence:\t {self.confidence}, \tbbox_shape: {self.bbox.shape}"
        if self.tracker_id is not None and self.tracker_id.shape == self.class_id.shape:
            s += f", \ttrack_ids:\t {self.tracker_id}"
        return s + ")"

    def json(self) -> dict:
        """
        Convert the Detections object to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the Detections object with the following keys:
            - "bbox" (list): The bounding box coordinates.
            - "confidence" (list): The confidence scores.
            - "class_id" (list): The class IDs.
            - "tracker_id" (list or None): The tracker IDs, or None if tracker_id is not set or its shape does not match.
        """
        return {
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence.tolist(),
            "class_id": self.class_id.tolist(),
            "tracker_id": (
                self.tracker_id.tolist()
                if self.tracker_id is not None and self.tracker_id.shape == self.class_id.shape
                else None
            ),
            "_roi_compensated": self._roi_compensated,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Detections":
        """
        Create a Detections object from a JSON-serializable dictionary.

        Args:
            data: A dictionary representation of the Detections object.

        Returns:
            An instance of the Detections class.
        """
        bbox = np.array(data["bbox"])
        confidence = np.array(data["confidence"])
        class_id = np.array(data["class_id"])
        tracker_id = np.array(data["tracker_id"]) if data.get("tracker_id") is not None else None

        instance = cls(bbox=bbox, confidence=confidence, class_id=class_id)
        if tracker_id is not None and len(tracker_id) > 0:
            instance.tracker_id = tracker_id
        instance._roi_compensated = data["_roi_compensated"]
        return instance

    # PROPERTIES
    @property
    def area(self) -> np.ndarray:
        """
        Array of shape (n,) the area of the bounding boxes of N detections
        """
        widths = self.bbox[:, 2] - self.bbox[:, 0]
        heights = self.bbox[:, 3] - self.bbox[:, 1]
        return widths * heights

    @property
    def bbox_width(self) -> np.ndarray:
        """
        Array of shape (n,) the width of the bounding boxes of N detections
        """
        return self.bbox[:, 2] - self.bbox[:, 0]

    @property
    def bbox_height(self) -> np.ndarray:
        """
        Array of shape (n,) the height of the bounding boxes of N detections
        """
        return self.bbox[:, 3] - self.bbox[:, 1]

    @property
    def center_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        A tuple containing two arrays (x_centers, y_centers),
        where each array contains the center coordinates of the bounding boxes.
        """
        return ((self.bbox[:, 0] + self.bbox[:, 2]) / 2, (self.bbox[:, 1] + self.bbox[:, 3]) / 2)


class Poses(Result):
    """
    Data class for pose estimation results.
    """

    n_detections: int  #: Number of detected pose
    confidence: np.ndarray  #: Confidence scores related to the detected poses
    keypoints: np.ndarray  #: Detected keypoint coordinates
    keypoint_scores: np.ndarray  #: Confidence scores related to the detected keypoints
    bbox: np.ndarray  #: Optional bounding box related to the detected poses

    def __init__(
        self,
        n_detections=0,
        confidence=np.empty((0,)),
        keypoints=np.empty((0,)),
        keypoint_scores=np.empty((0,)),
        bbox=None,
    ) -> None:
        self.n_detections = n_detections
        self.confidence = confidence
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores
        self._bbox = bbox
        self.tracker_id = None

        self._roi_compensated = False

    def compensate_for_roi(self, roi: ROI):
        """
        Compensate the keypoints and bounding boxes for the given ROI.

        Args:
            roi: The ROI (normalized - left, top, width, height) to compensate for.
        """
        if roi == (0, 0, 1, 1) or self._roi_compensated or self.keypoints.size == 0:
            return

        self.keypoints[:, :, 0] = roi[0] + self.keypoints[:, :, 0] * roi[2]
        self.keypoints[:, :, 1] = roi[1] + self.keypoints[:, :, 1] * roi[3]

        if self._bbox is not None:
            self._bbox[:, 0] = roi[0] + self._bbox[:, 0] * roi[2]
            self._bbox[:, 1] = roi[1] + self._bbox[:, 1] * roi[3]
            self._bbox[:, 2] = roi[0] + self._bbox[:, 2] * roi[2]
            self._bbox[:, 3] = roi[1] + self._bbox[:, 3] * roi[3]

        self._roi_compensated = True

    @property
    def bbox(self):
        """
        Get the bounding box corresponding to the pose detection.
        """
        if self._bbox is not None:
            return self._bbox
        elif self.n_detections == 0:
            return None
        else:
            # Create bounding box around the outer keypoints
            kpts = self.keypoints.reshape(self.n_detections, -1, 2)  # (N, 17, 2)
            mins = np.ma.min(kpts, axis=1).filled(0)  # (N, 2) equals 0 for invalid points
            maxs = np.ma.max(kpts, axis=1).filled(0)  # (N, 2) equals 0 for invalid points

            # [x1, y1, x2, y2] of N detections
            return np.column_stack([mins[:, 0], mins[:, 1], maxs[:, 0], maxs[:, 1]])

    @bbox.setter
    def bbox(self, value):
        self._bbox = value

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, np.ndarray, int]]:
        """
        To iterate over the detections. 
        """
        for i in range(len(self)):
            yield (
                self.keypoints[i],
                self.confidence[i],
                self.keypoint_scores[i],
                self.bbox[i],
                self.tracker_id[i],
            )

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "Poses":
        """
        Returns a new Detections object with the selected detections.

        Args:
            index: The index or indices of the detections to select.

        Returns:
            A new Detections object with the selected detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.n_detections = len(self.confidence[index])
        res.confidence = self.confidence[index]
        res.keypoints = self.keypoints[index]
        res.keypoint_scores = self.keypoint_scores[index] if self.keypoint_scores is not None else None
        if self._bbox is not None:
            res._bbox = self._bbox[index] if len(self._bbox) > 0 else None
        res.tracker_id = self.tracker_id[index] if self.tracker_id is not None else None
        return res

    def __str__(self) -> str:
        """
        Return a string representation of the Poses object.

        Returns:
            A string representation of the Poses object.
        """

        s = f"Poses(n_detections: {self.n_detections}, \tconfidence:\t {self.confidence}, \tkeypoints: {self.keypoints},"
        if self.bbox is not None and self.bbox.shape == self.confidence.shape:
            s += f", \tbbox:\t {self.bbox}"

        if self.tracker_id is not None and self.tracker_id.shape == self.confidence.shape:
            s += f", \ttrack_ids:\t {self.tracker_id}"
        return s + ")"

    def __len__(self):
        """
        Returns the number of detections.
        """
        return self.n_detections

    def __copy__(self) -> "Poses":
        """
        Returns a copy of the current detections.
        """
        new_instance = Poses()
        new_instance.confidence = np.copy(self.confidence)
        new_instance.keypoints = np.copy(self.keypoints)
        new_instance.keypoint_scores = np.copy(self.keypoint_scores)
        new_instance.tracker_id = np.copy(self.tracker_id)
        if self._bbox is not None:
            new_instance._bbox = np.copy(self._bbox)
        new_instance._roi_compensated = copy.deepcopy(self._roi_compensated)
        return new_instance

    def copy(self) -> "Poses":
        """
        Returns a copy of the current detections.
        """
        return self.__copy__()

    def json(self) -> dict:
        """
        Convert the Detections object to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the Detections object with the following keys:
            - "n_detections" (int): Number of detected poses.
            - "confidence" (list): Confidence scores related to the detected poses.
            - "keypoints" (list): Detected keypoint coordinates.
            - "keypoint_scores" (list): Confidence scores related to the detected keypoints.
            - "bbox" (list): The bounding box coordinates if available.
            - "tracker_id" (list): The tracker IDs if available.
        """
        return {
            "n_detections": self.n_detections,
            "confidence": self.confidence.tolist(),
            "keypoints": self.keypoints.tolist(),
            "keypoint_scores": self.keypoint_scores.tolist(),
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
            "tracker_id": (
                self.tracker_id.tolist()
                if self.tracker_id is not None and len(self.tracker_id) == self.n_detections
                else None
            ),
            "_roi_compensated": self._roi_compensated,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Poses":
        """
        Create a Poses instance from a JSON-serializable dictionary.

        Args:
            data: JSON-serializable dictionary with pose estimation data.

        Returns:
            The Poses instance created from the JSON data.
        """
        instance = cls(
            n_detections=data["n_detections"],
            confidence=np.array(data["confidence"], dtype=np.float32),
            keypoints=np.array(data["keypoints"], dtype=np.float32),
            keypoint_scores=np.array(data["keypoint_scores"], dtype=np.float32),
            bbox=np.array(data["bbox"], dtype=np.float32) if data.get("bbox") is not None else None,
        )
        tracker_id = data.get("tracker_id")
        if tracker_id is not None and len(tracker_id) > 0:
            instance.tracker_id = np.array(tracker_id)
        instance._roi_compensated = data["_roi_compensated"]
        return instance


class Segments(Result):
    """
    Data class for segmentation results.
    """

    mask: np.ndarray  #: 2D Mask arrays containing the id for each identified segment in the input tensor.

    def __init__(self, mask: np.ndarray = np.empty((0,))):
        self.mask = mask.astype(np.uint8)
        self._roi_compensated = False

    def compensate_for_roi(self, roi: ROI):
        """
        Compensate the segmentation mask for the given ROI.

        Args:
            roi: The ROI (normalized - left, top, width, height) to compensate for.
        """
        if roi == (0, 0, 1, 1) or self._roi_compensated:
            return

        if self.mask.size > 0:
            h, w = self.mask.shape
            start_h, start_w = int(roi[1] * h / roi[3]), int(roi[0] * w / roi[2])
            new_masks = np.zeros((int(h / roi[3]), int(w / roi[2])), dtype=self.mask.dtype)
            new_masks[start_h : start_h + h, start_w : start_w + w] = self.mask
            self.mask = new_masks

        self._roi_compensated = True

    @property
    def n_segments(self) -> int:
        """
        The number found segments, while ignore the background.
        """
        return len(self.indices)

    @property
    def indices(self) -> List[int]:
        """
        Found indices in the mask and ignore the background (id: 0).
        """
        found_indices = np.unique(self.mask)
        return found_indices[found_indices != 0]

    def get_mask(self, id: int) -> np.ndarray:
        """
        Returns the binary mask of a specific index.

        Args:
            id: The index of the mask to return.

        Returns:
            A numpy array of shape (h, w) representing the mask of the specified index.
        """
        return (self.mask == id).astype(np.uint8)

    def __str__(self) -> str:
        """
        Return a string representation of the Segments object.

        Returns:
            A string representation of the Segments object.
        """
        s = "Segments("
        if self.mask is not None:
            s += f"\tmask:\t {self.mask.shape}"
        return s + ")"

    def __copy__(self) -> "Segments":
        """
        Returns a copy of the current detections.
        """
        new_instance = Segments()
        new_instance.mask = np.copy(self.mask)
        new_instance._roi_compensated = copy.deepcopy(self._roi_compensated)
        return new_instance

    def copy(self) -> "Segments":
        """
        Returns a copy of the current detections.
        """
        return self.__copy__()

    def json(self) -> dict:
        """
        Convert the Segments object to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the Segments object with the following keys:
            - "n_segments" (int): Number of detected segments.
            - "indices" (list): List of the index corresponding to each segment.
            - "mask" (str): Mask array for each segment (compressed and base64 encoded).
            - "mask_shape" (tuple): The shape of the mask.
            - "_roi_compensated" (bool): Whether the ROI has been compensated for.
        """
        return {
            "n_segments": self.n_segments,
            "indices": self.indices.tolist(),
            "mask": base64.b64encode(gzip.compress(self.mask.astype(np.uint8).tobytes())).decode("utf-8"),
            "mask_shape": self.mask.shape,
            "_roi_compensated": self._roi_compensated,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Segments":
        """
        Create a Segments instance from a JSON-serializable dictionary.

        Args:
            data: JSON-serializable dictionary with segmentation data.

        Returns:
            The Segments instance created from the JSON data.
        """
        # Decode and decompress the mask data
        instance = cls(
            mask=np.frombuffer(gzip.decompress(base64.b64decode(data["mask"])), dtype=np.uint8).reshape(
                data["mask_shape"]
            )
        )
        instance._roi_compensated = data["_roi_compensated"]
        return instance

    def to_instance_segments(self, instance_args: object) -> "InstanceSegments":
        """
        Perform connected component analysis on a semantic segmentation mask to provide instance segmentation
        masks. Applies a watershed algorithm to CCA output to improve the results that are connected

        Args:
            instance_args: Arguments for instance segmentation.

        Returns:
            An InstanceSegments object with the instance segmentation masks, class ids.
        """
        if instance_args is None:
            raise ValueError("Input default 'instance_args' to run CCA")

        class_id = []
        instance_masks = []

        # Perform CCA and Watershed for each class label
        for class_label in self.indices:
            binary_mask = self.get_mask(class_label)

            # Label connected components in the binary mask using OpenCV
            num_labels, labeled_mask = cv2.connectedComponents(binary_mask)
            labeled_mask += 1
            E_kernel = np.ones((instance_args.erosion_kernel, instance_args.erosion_kernel), np.uint8)
            D_kernel = np.ones((instance_args.dilate_kernel, instance_args.dilate_kernel), np.uint8)
            opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, E_kernel, iterations=2)
            sure_bg = cv2.dilate(opening, D_kernel, iterations=instance_args.dilate_iteration)

            # Watershed algorithm
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # Distance transform
            dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)  # Normalize distance transform
            _, sure_fg = cv2.threshold(
                dist_transform, instance_args.dist_threshold * dist_transform.max(), 255, cv2.THRESH_BINARY
            )  # Threshold the distance transform to obtain markers
            sure_fg = cv2.erode(sure_fg.astype(np.uint8), E_kernel, iterations=instance_args.erosion_iteration)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)  # Perform connected components on the markers
            markers += 1
            markers[unknown == 255] = 0  # Mark the unknown regions (where binary_mask is 0) with 0
            markers = cv2.watershed(
                cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR), markers
            )  # Apply Watershed algorithm

            # Label the watershed regions
            labeled_mask = np.zeros_like(markers)
            labeled_mask[markers > 1] = markers[markers > 1]

            # Display output in config mode
            if instance_args.config_mode:
                cv2.imshow(f"fg-{class_label}", sure_fg)
                cv2.imshow(f"dist-{class_label}", dist_transform)

            # Filter out small sized masks
            for label_id in np.unique(labeled_mask):
                if label_id <= 1:
                    continue
                filtered_mask = np.zeros_like(labeled_mask)
                component_size = np.sum(labeled_mask == label_id)
                if component_size >= instance_args.size_threshold:
                    filtered_mask[labeled_mask == label_id] = 1

                    # Store the labeled mask in the arrays
                    instance_masks.append(filtered_mask)
                    class_id.append(int(class_label))

        return InstanceSegments(
            mask=np.array(instance_masks),
            class_id=np.array(class_id),
            confidence=np.ones_like(class_id),  # Dummy confidence
        )


class InstanceSegments(Result):
    """
    Data class for instance segmentation results.
    """

    mask: np.ndarray  #: Mask arrays containing the id for each identified segment.
    bbox: np.ndarray  #: Bounding boxes for each instance.
    class_id: np.ndarray  #: Class ids for each instance.
    confidence: np.ndarray  #: Confidence scores for each instance.
    tracker_id: np.ndarray  #: Tracker ids for each instance.

    def __init__(self, mask: np.ndarray = np.empty((0,)), bbox=None, confidence=None, class_id=None):
        self.mask = mask.astype(np.uint8)
        self._bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.tracker_id = None

        self._roi_compensated = False

    def compensate_for_roi(self, roi: ROI):
        """
        Compensate the instance segmentation masks and optional bounding boxes for the given ROI.

        Args:
            roi: The ROI (normalized - left, top, width, height) to compensate for.
        """
        if roi == (0, 0, 1, 1) or self._roi_compensated:
            return

        # initialize as background values and place mask(s) at ROI offset
        if self.mask.size > 0:
            n, h, w = self.mask.shape
            start_h, start_w = int(roi[1] * h / roi[3]), int(roi[0] * w / roi[2])
            new_masks = np.zeros((n, int(h / roi[3]), int(w / roi[2])), dtype=self.mask.dtype)
            new_masks[:, start_h : start_h + h, start_w : start_w + w] = self.mask
            self.mask = new_masks

        if self._bbox is not None:
            self._bbox[:, 0] = roi[0] + self._bbox[:, 0] * roi[2]
            self._bbox[:, 1] = roi[1] + self._bbox[:, 1] * roi[3]
            self._bbox[:, 2] = roi[0] + self._bbox[:, 2] * roi[2]
            self._bbox[:, 3] = roi[1] + self._bbox[:, 3] * roi[3]

        self._roi_compensated = True

    @property
    def n_segments(self) -> int:
        """
        The number segmented instances.
        """
        return len(self.indices)

    @property
    def indices(self) -> List[int]:
        """
        Segmented instances ids.
        """
        return self.class_id

    def oriented_bbox(self) -> np.ndarray:
        """
        Calculate oriented bounding boxes for each instance mask. Can't be used in tracker

        Returns:
            A numpy array (N, 4, 2) containing the bounding box corner points for each instance.
            Each bounding box is represented as a 4x2 array of normalized corner coordinates (x, y) in the range [0, 1].
            Returns None if there are no masks.
        """
        if len(self.mask) == 0:
            return None
        else:
            oriented_bboxes = []
            _, height, width = self.mask.shape
            for mask in self.mask:
                # Find contours of the mask
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue

                # Calculate normalized oriented bounding box
                rect = cv2.minAreaRect(contours[0])
                box_points = cv2.boxPoints(rect)
                box_points[:, 0] /= width
                box_points[:, 1] /= height

                oriented_bboxes.append(box_points)
            return np.array(oriented_bboxes)

    @property
    def bbox(self) -> np.ndarray:
        """
        Calculate bounding boxes for each instance mask.

        Returns:
            A numpy array of shape (N, 4) containing the bounding boxes.
            For normal bounding boxes: (x, y, w, h)
        """

        if self._bbox is not None:
            return self._bbox
        elif len(self.mask) == 0:
            self._bbox = np.empty((0, 4))
            return self._bbox
        else:
            _bbox = []
            _, height, width = self.mask.shape
            for mask in self.mask:
                # Find contours of the mask
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue
                # Calculate normal bounding box
                x, y, w, h = cv2.boundingRect(contours[0])
                _bbox.append((x / width, y / height, (x + w) / width, (y + h) / height))
            self._bbox = np.array(_bbox)
            return self._bbox

    @bbox.setter
    def bbox(self, value):
        self._bbox = value

    def __len__(self):
        if self.mask is not None:
            return len(self.mask)
        else:
            return 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, np.ndarray, int]]:
        """
        Iterate over the instance segments.
        """
        for i in range(len(self)):
            yield (
                self.mask[i] if self.mask is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.bbox[i] if self.bbox is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    def __str__(self) -> str:
        """
        Return a string representation of the InstanceSegments object.

        Returns:
            A string representation of the InstanceSegments object.
        """
        s = "InstanceSegments("
        if self.class_id is not None:
            s += f"class_id:\t {self.class_id}, \tconfidence:\t {self.confidence}, \tbbox_shape: {self.bbox.shape}"
        if self.tracker_id is not None and self.tracker_id.shape == self.class_id.shape:
            s += f", \ttrack_ids:\t {self.tracker_id}"
        if self.mask is not None:
            s += f"\tmask:\t {self.mask.shape}"
        return s + ")"

    def __copy__(self):
        """
        Returns a copy of the current detections.
        """
        new_instance = InstanceSegments()
        new_instance.mask = np.copy(self.mask)
        new_instance.bbox = np.copy(self.bbox)
        new_instance.confidence = np.copy(self.confidence)
        new_instance.class_id = np.copy(self.class_id)
        new_instance.tracker_id = np.copy(self.tracker_id)
        new_instance._roi_compensated = copy.deepcopy(self._roi_compensated)
        return new_instance

    def copy(self):
        """
        Returns a copy of the current InstanceSegments.
        """
        return self.__copy__()

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "InstanceSegments":
        """
        Returns a new InstanceSegments object with the selected instance segment detections.

        Args:
            index: The index or indices of the instance segment detections to select.

        Returns:
            A new InstanceSegments object with the selected detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.mask = self.mask[index]
        res.confidence = self.confidence[index]
        res.class_id = self.class_id[index]
        res.bbox = self.bbox[index] if self.bbox is not None else None
        res.tracker_id = self.tracker_id[index] if self.tracker_id is not None else None
        return res

    def json(self) -> dict:
        """
        Convert the InstanceSegments object to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the InstanceSegments object with the following keys:
            - "n_segments" (int): Number of detected instance segments.
            - "indices" (list): List of the index corresponding to each instance segment.
            - "mask" (str): Mask array for each instance segment (compressed and base64 encoded).
            - "mask_shape" (tuple): The shape of the mask.
            - "bbox" (list): The bounding box coordinates.
            - "confidence" (list): The confidence scores.
            - "class_id" (list): The class IDs.
            - "tracker_id" (list or None): The tracker IDs, or None if tracker_id is not set or its shape does not match.
            - "_roi_compensated" (bool): Whether the ROI has been compensated for.
        """
        return {
            "n_segments": self.n_segments,
            "indices": self.indices.tolist(),
            "mask": base64.b64encode(gzip.compress(self.mask.tobytes())).decode("utf-8"),
            "mask_shape": self.mask.shape,
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence.tolist(),
            "class_id": self.class_id.tolist(),
            "tracker_id": (
                self.tracker_id.tolist()
                if self.tracker_id is not None and self.tracker_id.shape == self.class_id.shape
                else None
            ),
            "_roi_compensated": self._roi_compensated,
        }

    @classmethod
    def from_json(cls, data: dict) -> "InstanceSegments":
        """
        Create a InstanceSegments instance from a JSON-serializable dictionary.

        Args:
            data: JSON-serializable dictionary with instance segmentation data.

        Returns:
            The InstanceSegments instance created from the JSON data.
        """
        bbox = np.array(data["bbox"])
        confidence = np.array(data["confidence"])
        class_id = np.array(data["class_id"])
        tracker_id = np.array(data["tracker_id"]) if data.get("tracker_id") is not None else None

        # Decode and decompress the mask data
        instance = cls(
            mask=np.frombuffer(gzip.decompress(base64.b64decode(data["mask"])), dtype=np.uint8).reshape(
                data["mask_shape"]
            ),
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
        )
        if tracker_id is not None and len(tracker_id) > 0:
            instance.tracker_id = tracker_id
        instance._roi_compensated = data["_roi_compensated"]
        return instance


class Anomaly(Result):
    """
    Data class for anomaly detection results.
    """

    score: float  #: The anomaly score indicating the likelihood of an anomaly on the full detection.
    heatmap: np.ndarray  #: A 2D grid representing the anomaly score heatmap on the frame.

    def __init__(self, score: float = 0.0, heatmap=np.empty((0,))) -> None:
        self.score = float(score)
        self.heatmap = heatmap.astype(np.float32)

        self._roi_compensated = False

    def compensate_for_roi(self, roi: ROI):
        """
        Compensate the anomaly score heatmap for the given ROI.

        Args:
            roi: The ROI (normalized - left, top, width, height) to compensate for.
        """
        if roi == (0, 0, 1, 1) or self._roi_compensated:
            return

        if self.heatmap.size > 0:
            h, w = self.heatmap.shape
            new_heatmap = np.zeros((int(h / roi[3]), int(w / roi[2])), dtype=self.heatmap.dtype)
            start_h, start_w = int(roi[1] * h / roi[3]), int(roi[0] * w / roi[2])
            new_heatmap[start_h : start_h + h, start_w : start_w + w] = self.heatmap
            self.heatmap = new_heatmap

        self._roi_compensated = True

    def get_mask(self, score_threshold: float, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Returns the mask with the specified color.

        Args:
            score_threshold: The threshold to apply to the heatmap.
            color: The BGR color to use for the mask. Default is red (0, 0, 255).

        Returns:
            A 3-channel (BGR) numpy array representing the colored mask
        """
        mask = np.zeros(self.heatmap.shape + (3,), dtype=np.uint8)
        mask[self.heatmap >= score_threshold, :] = color
        return mask

    def __str__(self) -> str:
        """
        Return a string representation of the Anomaly object.

        Returns:
            A string representation of the Anomaly object.
        """
        heatmap_str = np.array2string(self.heatmap, threshold=10, edgeitems=2)
        return f"Anomaly(score: {self.score}, heatmap: \n{heatmap_str})"

    def json(self) -> dict:
        """
        Convert the Anomaly object to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the Anomaly object with the following keys:
            - "score" (float): The anomaly score.
            - "heatmap" (str): The anomaly score heatmap (compressed and base64 encoded).
            - "heatmap_shape" (tuple): The shape of the heatmap.
        """
        return {
            "score": self.score,
            "heatmap": base64.b64encode(gzip.compress(self.heatmap.tobytes())).decode("utf-8"),
            "heatmap_shape": self.heatmap.shape,
            "_roi_compensated": self._roi_compensated,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Anomaly":
        """
        Create an Anomaly instance from a JSON-serializable dictionary.

        Args:
            data: JSON-serializable dictionary with anomaly detection data.

        Returns:
            The Anomaly instance created from the JSON data.
        """
        # Decode and decompress the heatmap data
        instance = cls(
            score=data["score"],
            heatmap=np.frombuffer(gzip.decompress(base64.b64decode(data["heatmap"])), dtype=np.float32).reshape(
                data["heatmap_shape"]
            ),
        )
        instance._roi_compensated = data["_roi_compensated"]
        return instance
