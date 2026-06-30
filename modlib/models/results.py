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
import copy
import gzip
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Iterator, List, Optional, Sequence, Tuple, Union

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

    def _as_roi(self, other) -> "ROI":
        if isinstance(other, ROI):
            return other
        elif isinstance(other, (tuple, list)) and len(other) == 4:
            return ROI(left=float(other[0]), top=float(other[1]), width=float(other[2]), height=float(other[3]))
        else:
            raise ValueError(f"Invalid type or format: {type(other)}, {other}")

    def __eq__(self, other: "ROI") -> bool:
        other = self._as_roi(other)
        return (
            self.left == other.left and
            self.top == other.top and
            self.width == other.width and
            self.height == other.height
        )

    def __ne__(self, other: "ROI") -> bool:
        return not self == other


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
        self.bbox = np.clip(self.bbox, 0, 1)

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
        new_instance.tracker_id = np.copy(self.tracker_id) if self.tracker_id is not None else None
        new_instance._roi_compensated = copy.copy(self._roi_compensated)
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
        self.keypoints = np.clip(self.keypoints, 0, 1)

        if self._bbox is not None:
            self._bbox[:, 0] = roi[0] + self._bbox[:, 0] * roi[2]
            self._bbox[:, 1] = roi[1] + self._bbox[:, 1] * roi[3]
            self._bbox[:, 2] = roi[0] + self._bbox[:, 2] * roi[2]
            self._bbox[:, 3] = roi[1] + self._bbox[:, 3] * roi[3]
            self._bbox = np.clip(self._bbox, 0, 1)

        self._roi_compensated = True

    @property
    def bbox(self):
        """
        Get the bounding box corresponding to the pose detection.
        """
        if self._bbox is not None:
            return self._bbox
        elif self.n_detections == 0:
            return np.empty((0, 4))
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
                self.tracker_id[i] if self.tracker_id is not None else None,
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
        new_instance.n_detections = self.n_detections
        new_instance.confidence = np.copy(self.confidence)
        new_instance.keypoints = np.copy(self.keypoints)
        new_instance.keypoint_scores = np.copy(self.keypoint_scores)
        new_instance.tracker_id = np.copy(self.tracker_id) if self.tracker_id is not None else None
        if self._bbox is not None:
            new_instance._bbox = np.copy(self._bbox)
        new_instance._roi_compensated = copy.copy(self._roi_compensated)
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

    mask: np.ndarray  #: 2D Mask arrays containing the id for each identified segment in the input tensor. Mask background pixels are represented by -1 (/255) for the uint8 array.

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
            out_h, out_w = int(h / roi[3]), int(w / roi[2])
            h_start, w_start = int(roi[1] * h / roi[3]), int(roi[0] * w / roi[2])

            # starting offsets for resulting and input masks
            start_h, start_w = max(0, -h_start), max(0, -w_start)
            out_start_h, out_start_w = max(0, h_start), max(0, w_start)
            delta_h = min(h - start_h, out_h - out_start_h)
            delta_w = min(w - start_w, out_w - out_start_w)

            # create compensated output mask
            new_masks = np.full((out_h, out_w), 255, dtype=self.mask.dtype)
            new_masks[out_start_h : out_start_h + delta_h, out_start_w : out_start_w + delta_w] = self.mask[
                start_h : start_h + delta_h, start_w : start_w + delta_w
            ]
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
        Found indices in the mask and ignore the background (id: 255).
        """
        found_indices = np.unique(self.mask)
        return found_indices[found_indices != 255]

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
        new_instance._roi_compensated = copy.copy(self._roi_compensated)
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
            - "mask" (str): Mask array for each segment (uint8 byte-array, compressed and base64 encoded).
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

    Instance masks are stored internally as bbox-cropped 2D uint8 arrays for memory efficiency.
    """

    bbox: np.ndarray  #: Bounding boxes for each instance (normalized `x1, y1, x2, y2`).
    class_id: np.ndarray  #: Class ids for each instance.
    confidence: np.ndarray  #: Confidence scores for each instance.
    tracker_id: np.ndarray  #: Tracker ids for each instance.
    mask_shape: Tuple[int, int]  #: Full canvas shape `(H, W)`.

    def __init__(
        self,
        mask: Union[np.ndarray, List[np.ndarray]],
        confidence: np.ndarray,
        class_id: np.ndarray,
        bbox: Optional[np.ndarray] = None,
        mask_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the InstanceSegments object.

        Args:
            mask: Either a dense `(N, H, W)` stack (converted to cropped storage), or a list of
                cropped 2D masks per instance (requires `bbox` and `mask_shape`).
            confidence: Per-instance confidence scores.
            class_id: Per-instance class ids.
            bbox: `(N, 4)` normalized `x1, y1, x2, y2`. Required when `mask` is a list of crops.
            mask_shape: Full canvas shape `(H, W)`. Required when `mask` is a list of crops.
        """
        self.tracker_id = None
        self._roi_compensated = False

        # Empty default / explicit empty
        if (isinstance(mask, np.ndarray) and mask.size == 0) or (isinstance(mask, list) and len(mask) == 0):
            self._mask_crops = []
            self.mask_shape = mask_shape if mask_shape is not None else (0, 0)
            self._bbox = np.empty((0, 4), dtype=np.float32)
            self.class_id = np.empty((0,), dtype=np.int32)
            self.confidence = np.empty((0,), dtype=np.float32)
            return

        if isinstance(mask, list):
            if bbox is None or mask_shape is None:
                raise ValueError("When mask is a list of cropped arrays, both bbox and mask_shape are required.")
            self._mask_crops = [np.asarray(m, dtype=np.uint8, order="C") for m in mask]
            self.mask_shape = (int(mask_shape[0]), int(mask_shape[1]))
            self._bbox = np.asarray(bbox, dtype=np.float32)
            if self._bbox.shape[0] != len(self._mask_crops):
                raise ValueError("bbox length must match number of cropped masks.")
            self.class_id = np.asarray(class_id, dtype=np.int32)
            self.confidence = np.asarray(confidence, dtype=np.float32)
            return

        elif isinstance(mask, np.ndarray):
            dense = mask.astype(np.uint8, copy=False)
            if dense.ndim == 2:
                dense = dense[None, ...]
            if dense.ndim != 3:
                raise ValueError(f"Dense mask must have shape (N, H, W), got {dense.shape}")

            self.mask_shape = (int(dense.shape[1]), int(dense.shape[2]))
            bbox_arr = np.asarray(bbox, dtype=np.float32) if bbox is not None else None
            if bbox_arr is None or bbox_arr.shape != (dense.shape[0], 4):
                bbox_arr = InstanceSegments._bbox_from_dense_masks(dense)
            self._bbox = bbox_arr.astype(np.float32, copy=False)
            self._mask_crops = InstanceSegments.compress_mask(dense, self._bbox)
            self.class_id = np.asarray(class_id, dtype=np.int32)
            self.confidence = np.asarray(confidence, dtype=np.float32)

        else:
            raise TypeError(f"mask must be ndarray or list of ndarrays, got {type(mask)}")

    @staticmethod
    def _bbox_to_slices(b: np.ndarray, height: int, width: int) -> Tuple[int, int, int, int]:
        """Convert normalized `x1,y1,x2,y2` to integer slice bounds `y1,y2,x1,x2` inclusive-exclusive."""
        x1, y1, x2, y2 = float(b[0]) * width, float(b[1]) * height, float(b[2]) * width, float(b[3]) * height
        xi1 = int(np.clip(np.floor(x1), 0, width))
        yi1 = int(np.clip(np.floor(y1), 0, height))
        xi2 = int(np.clip(np.ceil(x2), 0, width))
        yi2 = int(np.clip(np.ceil(y2), 0, height))
        if xi2 <= xi1:
            xi2 = min(width, xi1 + 1)
        if yi2 <= yi1:
            yi2 = min(height, yi1 + 1)
        return yi1, yi2, xi1, xi2

    def compress_mask(mask: np.ndarray, bbox: np.ndarray) -> List[np.ndarray]:
        """
        Crop each instance mask to the corresponding normalized bounding box region.

        Args:
            mask: Dense masks of shape `(N, H, W)` (uint8).
            bbox: Array of shape `(N, 4)` with normalized `x1, y1, x2, y2`.

        Returns:
            List of length `N` of cropped 2D uint8 masks.
        """
        if mask.ndim != 3:
            raise ValueError(f"compress_mask expects mask of shape (N, H, W), got {mask.shape}")
        n, height, width = mask.shape
        bbox = np.asarray(bbox, dtype=np.float32)
        if bbox.shape != (n, 4):
            raise ValueError(f"bbox must have shape ({n}, 4), got {bbox.shape}")
        mask_u8 = mask.astype(np.uint8, copy=False)
        crops: List[np.ndarray] = []
        for i in range(n):
            yi1, yi2, xi1, xi2 = InstanceSegments._bbox_to_slices(bbox[i], height, width)
            crops.append(np.ascontiguousarray(mask_u8[i, yi1:yi2, xi1:xi2]))
        return crops

    @staticmethod
    def decompress_mask(masks: Sequence[np.ndarray], bbox: np.ndarray, mask_shape: Tuple[int, int]) -> np.ndarray:
        """
        Place cropped instance masks back onto a dense `(N, H, W)` canvas.

        Args:
            masks: Sequence of `N` cropped 2D uint8 masks (must match `compress_mask` geometry).
            bbox: `(N, 4)` normalized `x1, y1, x2, y2` used when cropping.
            mask_shape: Full canvas shape `(H, W)`.

        Returns:
            Dense array of shape `(N, H, W)` uint8.
        """
        height, width = int(mask_shape[0]), int(mask_shape[1])
        bbox = np.asarray(bbox, dtype=np.float32)
        n = bbox.shape[0]
        if len(masks) != n:
            raise ValueError(f"Expected {n} cropped masks, got {len(masks)}")
        out = np.zeros((n, height, width), dtype=np.uint8)
        for i in range(n):
            yi1, yi2, xi1, xi2 = InstanceSegments._bbox_to_slices(bbox[i], height, width)
            crop = np.asarray(masks[i], dtype=np.uint8)
            ch, cw = yi2 - yi1, xi2 - xi1
            if crop.shape == (ch, cw):
                out[i, yi1:yi2, xi1:xi2] = crop
            else:
                raise ValueError(f"Cropped mask shape {crop.shape} does not match bbox slice ({ch}, {cw}) for instance {i}")
        return out

    @staticmethod
    def _bbox_from_dense_masks(mask: np.ndarray) -> np.ndarray:
        """Infer normalized `x1,y1,x2,y2` boxes from dense instance masks."""
        _, height, width = mask.shape
        rows = []
        for m in mask.astype(np.uint8, copy=False):
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                rows.append((0.0, 0.0, 0.0, 0.0))
            else:
                x, y, w, h = cv2.boundingRect(contours[0])
                rows.append((x / width, y / height, (x + w) / width, (y + h) / height))
        return np.asarray(rows, dtype=np.float32)

    @property
    def mask_crops(self) -> List[np.ndarray]:
        """Internal per-instance cropped masks (read-only copy of references; do not mutate in place)."""
        return self._mask_crops

    @property
    def mask(self) -> np.ndarray:
        """
        Dense instance masks of shape `(N, H, W)` reconstructed from cropped storage.

        Note: This allocates a full `(N, H, W)` array when called.
        """
        if len(self._mask_crops) == 0:
            return np.empty((0,), dtype=np.uint8)
        return InstanceSegments.decompress_mask(self._mask_crops, self._bbox, self.mask_shape)

    def compensate_for_roi(self, roi: ROI):
        """
        Compensate the instance segmentation masks and optional bounding boxes for the given ROI.

        Args:
            roi: The ROI (normalized - left, top, width, height) to compensate for.
        """
        if roi == (0, 0, 1, 1) or self._roi_compensated:
            return

        if len(self._mask_crops) > 0 and self._bbox is not None:
            h, w = self.mask_shape
            out_h, out_w = int(h / roi[3]), int(w / roi[2])
            h_start, w_start = int(roi[1] * h / roi[3]), int(roi[0] * w / roi[2])

            slices = np.asarray([InstanceSegments._bbox_to_slices(b, h, w) for b in self._bbox], dtype=np.int32)
            by1 = slices[:, 0] + h_start
            by2 = slices[:, 1] + h_start
            bx1 = slices[:, 2] + w_start
            bx2 = slices[:, 3] + w_start

            # Keep coords inside slice edges so floor/ceil in _bbox_to_slices reproduces exact ints.
            eps = 1e-4
            new_bbox = np.empty_like(self._bbox, dtype=np.float32)
            new_bbox[:, 0] = np.where(bx1 <= 0, 0.0, (bx1 + eps) / out_w)
            new_bbox[:, 1] = np.where(by1 <= 0, 0.0, (by1 + eps) / out_h)
            new_bbox[:, 2] = np.where(bx2 >= out_w, 1.0, (bx2 - eps) / out_w)
            new_bbox[:, 3] = np.where(by2 >= out_h, 1.0, (by2 - eps) / out_h)

            self.mask_shape = (out_h, out_w)
            self._bbox = new_bbox.astype(np.float32, copy=False)

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

    def oriented_bbox(self) -> "OBB":
        """
        Calculate oriented bounding boxes for each instance mask.

        Returns:
            An OBB object with the oriented bounding boxes of the instance segments.
        """
        if len(self._mask_crops) == 0:
            return OBB()

        height, width = self.mask_shape
        obb_bboxes = []
        obb_angles = []
        valid_indices = []

        for idx, crop in enumerate(self._mask_crops):
            contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            contour = max(contours, key=cv2.contourArea)
            (cx, cy), (w_rect, h_rect), angle = cv2.minAreaRect(contour)
            a = angle * np.pi / 180.0

            yi1, yi2, xi1, xi2 = InstanceSegments._bbox_to_slices(self._bbox[idx], height, width)
            gx = xi1 + cx
            gy = yi1 + cy

            obb_bboxes.append((gx / width, gy / height, w_rect / width, h_rect / height))
            obb_angles.append(a)
            valid_indices.append(idx)

        if len(valid_indices) == 0:
            return OBB()

        obb = OBB(
            bbox=np.asarray(obb_bboxes, dtype=np.float32),
            confidence=np.asarray(self.confidence)[valid_indices],
            class_id=np.asarray(self.class_id)[valid_indices],
            angle=np.asarray(obb_angles, dtype=np.float32),
        )
        if self.tracker_id is not None:
            obb.tracker_id = np.asarray(self.tracker_id)[valid_indices]
        obb._roi_compensated = self._roi_compensated

        return obb

    @property
    def bbox(self) -> np.ndarray:
        """
        Bounding boxes for each instance.

        Returns:
            A numpy array of shape `(N, 4)` with normalized `x1, y1, x2, y2`.
        """

        if len(self._mask_crops) == 0:
            self._bbox = np.empty((0, 4), dtype=np.float32)
            return self._bbox
        if self._bbox is not None:
            return self._bbox
        raise RuntimeError("InstanceSegments has masks but bbox is unset; provide bbox when using cropped masks.")

    @bbox.setter
    def bbox(self, value):
        self._bbox = np.asarray(value, dtype=np.float32) if value is not None else None

    def __len__(self):
        return len(self._mask_crops)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, int, np.ndarray, int]]:
        """
        Iterate over the instance segments.
        """
        dense = self.mask
        for i in range(len(self)):
            yield (
                dense[i],
                self.class_id[i] if self.class_id is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.bbox[i] if self._bbox is not None and len(self._bbox) > i else None,
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
        n = len(self._mask_crops)
        if n > 0:
            s += f"\tmask:\t (n={n}, *{self.mask_shape})"
        else:
            s += "\tmask:\t empty"
        return s + ")"

    def __copy__(self):
        """
        Returns a copy of the current detections.
        """
        new_instance = InstanceSegments.__new__(InstanceSegments)
        new_instance._mask_crops = [np.copy(c) for c in self._mask_crops]
        new_instance.mask_shape = self.mask_shape
        new_instance._bbox = np.copy(self._bbox) if self._bbox is not None else None
        new_instance.class_id = np.copy(self.class_id)
        new_instance.confidence = np.copy(self.confidence)
        new_instance.tracker_id = np.copy(self.tracker_id) if self.tracker_id is not None else None
        new_instance._roi_compensated = copy.copy(self._roi_compensated)
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
            idxs = [index]
        elif isinstance(index, slice):
            idxs = list(range(len(self._mask_crops)))[index]
        elif isinstance(index, np.ndarray):
            idxs = np.flatnonzero(index).tolist()
        elif isinstance(index, tuple):
            idxs = list(index)
        else:
            idxs = list(index)

        new_instance = InstanceSegments.__new__(InstanceSegments)
        new_instance._mask_crops = [self._mask_crops[i] for i in idxs]
        new_instance.mask_shape = self.mask_shape
        new_instance._bbox = self._bbox[idxs] if self._bbox is not None else None
        new_instance.class_id = self.class_id[idxs]
        new_instance.confidence = self.confidence[idxs]
        new_instance.tracker_id = self.tracker_id[idxs] if self.tracker_id is not None else None
        new_instance._roi_compensated = copy.copy(self._roi_compensated)
        return new_instance

    def json(self) -> dict:
        """
        Convert the InstanceSegments object to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the InstanceSegments object with the following keys:
            - "n_segments" (int): Number of detected segments.
            - "mask_shape" (tuple): The shape of the mask.
            - "mask_crops" (list): List of cropped masks (base64 encoded and gzip compressed).
            - "bbox" (list): The bounding box coordinates.
            - "confidence" (list): The confidence scores.
            - "class_id" (list): The class ids.
            - "tracker_id" (list): The tracker ids.
            - "_roi_compensated" (bool): Whether the ROI has been compensated for.
        """
        return {
            "n_segments": self.n_segments,
            "mask_shape": list(self.mask_shape),
            "mask_crops": [
                base64.b64encode(gzip.compress(np.ascontiguousarray(c).tobytes())).decode("utf-8") for c in self._mask_crops
            ],
            "bbox": self.bbox.tolist() if self._bbox is not None else [],
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
        bbox = np.array(data["bbox"], dtype=np.float32)
        confidence = np.array(data["confidence"], dtype=np.float32)
        class_id = np.array(data["class_id"])
        tracker_id = np.array(data["tracker_id"]) if data.get("tracker_id") is not None else None
        mask_shape = (int(data["mask_shape"][0]), int(data["mask_shape"][1]))

        crops_decoded: List[np.ndarray] = []
        h, w = mask_shape
        for i, entry in enumerate(data["mask_crops"]):
            yi1, yi2, xi1, xi2 = cls._bbox_to_slices(bbox[i], h, w)
            flat = gzip.decompress(base64.b64decode(entry))
            crop = np.frombuffer(flat, dtype=np.uint8).reshape((yi2 - yi1, xi2 - xi1))
            crops_decoded.append(np.ascontiguousarray(crop))

        instance = cls(mask=crops_decoded, bbox=bbox, confidence=confidence, class_id=class_id, mask_shape=mask_shape)
        if tracker_id is not None and len(tracker_id) > 0:
            instance.tracker_id = tracker_id
        instance._roi_compensated = bool(data["_roi_compensated"])
        return instance

    def to_segments(self) -> "Segments":
        """
        Convert instance segmentation masks to a simple 2D semantic segmentation mask.
        If label_mapping is provided, map the class_ids to the new labels.
        For each pixel, selects the instance with maximum confidence among all instances
        covering that pixel, and assigns that instance's class_id to the output mask.

        Returns:
            A Segments object with a 2D mask containing class_ids for each pixel.
        """
        if len(self._mask_crops) == 0:
            return Segments(mask=np.empty((0,)))

        dense = InstanceSegments.decompress_mask(self._mask_crops, self._bbox, self.mask_shape)
        weighted = dense.astype(np.float32) * self.confidence[:, None, None]
        best_instance_idx = np.argmax(weighted, axis=0)
        any_mask = np.any(dense > 0, axis=0)
        output_mask = np.where(any_mask, self.class_id[best_instance_idx], -1).astype(np.uint8)

        segments = Segments(mask=output_mask)
        segments._roi_compensated = self._roi_compensated
        return segments


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
            - "_roi_compensated" (bool): Whether the ROI has been compensated for.
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



class OBB(Result):
    """
    Data class for oriented bounding box (OBB) detections.
    """

    bbox: np.ndarray  #: Array of shape (n, 4) the bounding boxes [x, y, w, h] of N detections (xy are center points)
    confidence: np.ndarray  #: Array of shape (n,) the confidence of N detections
    class_id: np.ndarray  #: Array of shape (n,) the class id of N detections
    angle: np.ndarray  #: Array of shape (n,) the angle of N detections
    tracker_id: np.ndarray  #: Array of shape (n,) the tracker id of N detections

    def __init__(self,
        bbox: np.ndarray = np.empty((0, 4)),
        confidence: np.ndarray = np.empty((0,)),
        class_id: np.ndarray = np.empty((0,)),
        angle: np.ndarray = np.empty((0,)),
    ) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.angle = angle
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

        ux = np.cos(self.angle) * roi[2]
        uy = np.sin(self.angle) * roi[3]
        vx = -np.sin(self.angle) * roi[2]
        vy = np.cos(self.angle) * roi[3]

        self.angle = np.arctan2(uy, ux).astype(np.float32)
        self.bbox[:, 0] = roi[0] + self.bbox[:, 0] * roi[2]
        self.bbox[:, 1] = roi[1] + self.bbox[:, 1] * roi[3]
        self.bbox[:, 2] = self.bbox[:, 2] * np.sqrt(ux**2 + uy**2)
        self.bbox[:, 3] = self.bbox[:, 3] * np.sqrt(vx**2 + vy**2)
        self.bbox = np.clip(self.bbox, 0, 1)

        self._roi_compensated = True

    def json(self) -> dict:
        """
        Convert the OBB object to a JSON-serializable dictionary.

        Returns:
            A dictionary representation of the OBB object with the following keys:
            - "bbox" (list): The oriented bounding boxes [x, y, w, h].
            - "confidence" (list): The confidence scores.
            - "class_id" (list): The class IDs.
            - "angle" (list): The orientation angles in radians.
            - "tracker_id" (list or None): The tracker IDs, or None if tracker_id is not set or its shape does not match.
            - "_roi_compensated" (bool): Whether the ROI has been compensated for.
        """
        return {
            "bbox": self.bbox.tolist(),
            "confidence": self.confidence.tolist(),
            "class_id": self.class_id.tolist(),
            "angle": self.angle.tolist(),
            "tracker_id": (
                self.tracker_id.tolist()
                if self.tracker_id is not None and self.tracker_id.shape == self.class_id.shape
                else None
            ),
            "_roi_compensated": self._roi_compensated,
        }

    @classmethod
    def from_json(cls, data: dict) -> "OBB":
        """
        Create an OBB object from a JSON-serializable dictionary.

        Args:
            data: A dictionary representation of the OBB object.

        Returns:
            An instance of the OBB class.
        """
        bbox = np.array(data["bbox"])
        confidence = np.array(data["confidence"])
        class_id = np.array(data["class_id"])
        angle = np.array(data["angle"])
        tracker_id = np.array(data["tracker_id"]) if data.get("tracker_id") is not None else None

        instance = cls(
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            angle=angle,
        )
        if tracker_id is not None and len(tracker_id) > 0:
            instance.tracker_id = tracker_id
        instance._roi_compensated = data["_roi_compensated"]
        return instance

    def __len__(self) -> int:
        """
        Returns the number of OBB detections.
        """
        return len(self.class_id)

    def __copy__(self) -> "OBB":
        """
        Returns a copy of the current OBB detections.
        """
        new_instance = OBB()
        new_instance.bbox = np.copy(self.bbox)
        new_instance.confidence = np.copy(self.confidence)
        new_instance.class_id = np.copy(self.class_id)
        new_instance.angle = np.copy(self.angle)
        new_instance.tracker_id = np.copy(self.tracker_id) if self.tracker_id is not None else None
        new_instance._roi_compensated = copy.copy(self._roi_compensated)
        return new_instance

    def copy(self) -> "OBB":
        """
        Returns a copy of the current OBB detections.
        """
        return self.__copy__()

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "OBB":
        """
        Returns a new OBB object with the selected OBB detections.

        Args:
            index: The index or indices of the OBB detections to select.

        Returns:
            A new OBB object with the selected OBB detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.confidence = self.confidence[index]
        res.class_id = self.class_id[index]
        res.bbox = self.bbox[index] if self.bbox is not None else None
        res.angle = self.angle[index] if self.angle is not None else None
        res.tracker_id = self.tracker_id[index] if self.tracker_id is not None else None
        return res

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, float, int]]:
        """
        To iterate over the OBB detections.
        """
        for i in range(len(self)):
            yield (
                self.bbox[i],
                self.confidence[i],
                self.class_id[i],
                self.angle[i],
                self.tracker_id[i] if self.tracker_id is not None else None,
            )
