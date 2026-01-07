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

from collections import deque
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np

from modlib.models.results import Detections


@dataclass
class MatcherSettings:
    """
    Helper dataclass containing the matcher settings.
    """

    MAX_MISSING_OVERLAP: int
    MAX_MISSING_TRACKER: int
    MIN_OVERLAP_THRESHOLD: float
    HYSTERESIS: float


class FIFOQueue:
    """
    The FIFOQueue class provides positional filtering, enabling smoother matching of objects.
    """

    def __init__(self, n, initial_bbox):
        self.max_length = n
        self.bbox = np.tile(initial_bbox, (n, 1))
        self.current_index = 0

    def push(self, item):
        self.bbox[:-1] = self.bbox[1:]
        self.bbox[-1] = item
        self.current_index = (self.current_index + 1) % self.max_length

    def pop(self):
        popped_item = self.bbox[0]
        self.bbox[:-1] = self.bbox[1:]
        self.bbox[-1] = 0
        return popped_item

    def get_average(self):
        return np.mean(self.bbox, axis=0)


class DetectedObject:
    """
    A class encapsulating the state information of each detected object,
    maintaining details such as position, state of overlap, and other relevant attributes.
    """

    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        tracker_id: float,
        overlap: float,
        settings: MatcherSettings,
    ):
        """
        Args:
            bbox: array of base object bbox
            confidence: confidence value of base object detection
            class_id: id value of the base detection
            tracker_id: tracker id result if being used. Will use a default number if not in use
            overlap: overlap_ratio of the objects being matched
        """
        self.bbox = bbox
        self.bboxes = FIFOQueue(8, bbox)
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.overlap = overlap
        self.missing_tracker_counter = 0
        self.overlap_list = deque([overlap] * settings.MAX_MISSING_OVERLAP)
        self.avg_overlap = 0
        self.overlapped = False
        self.uptime = 0
        self.s = settings

    def update(self, source: "DetectedObject"):
        """
        Update the values of object depending on changes to the objects being matched

        Args:
            source: Class with updated values
        """
        self.uptime += 1
        self.missing_tracker_counter = 0
        self.bboxes.push(source.bbox)
        self.bbox = self.bboxes.get_average()
        self.confidence = source.confidence

        if source.overlap > self.s.MIN_OVERLAP_THRESHOLD:
            self.overlap_list.clear()
            self.overlap_list.extend([source.overlap] * (self.s.MAX_MISSING_OVERLAP - 1))

        self.overlap_list.append(source.overlap)
        if len(self.overlap_list) > self.s.MAX_MISSING_OVERLAP:
            self.overlap_list.popleft()

    def missing(self) -> bool:
        """
        Increases the missing counter if object hasn't been detected in MAX_MISSING_TRACKER

        Returns:
            A bool value if counter is more than MAX_MISSING_TRACKER
        """
        self.missing_tracker_counter += 1
        return self.missing_tracker_counter > self.s.MAX_MISSING_TRACKER

    def get(self):
        self.avg_overlap = 0
        if len(self.overlap_list):
            self.avg_overlap = sum(self.overlap_list) / len(self.overlap_list)

        if self.avg_overlap > self.s.MIN_OVERLAP_THRESHOLD + self.s.HYSTERESIS / 2:
            self.overlapped = True
        elif self.avg_overlap < self.s.MIN_OVERLAP_THRESHOLD - self.s.HYSTERESIS / 2:
            self.overlapped = False

        return (
            self.bbox,
            self.avg_overlap,
            self.class_id,
            self.tracker_id,
            self.overlapped,
            self.uptime,
            self.missing_tracker_counter,
        )

    def __eq__(self, other):
        if isinstance(other, DetectedObject):
            return self.tracker_id == other.tracker_id
        return False

    def is_new(self):
        if self.uptime == 0:
            return True
        return False

    def __str__(self):
        return (
            f"bbox {self.bbox}\n"
            f"confidence {self.confidence}\n"
            f"class_id {self.class_id}\n"
            f"tracker_id {self.tracker_id}\n"
            f"overlap {self.overlap}\n"
        )


class Matcher:
    """
    The `Matcher` module is designed to evaluate spatial relationships between objects, such as determining whether one object is contained within another or whether two objects intersect. 
    It is suitable for both **simple relationships** (e.g., checking if one object is within another) and **complex relationships** (e.g., evaluating overlaps between multiple objects across different classes).  
    
    For example, to create and use Matcher it can be used like this:
    ```
    from modlib.apps import Matcher

    matcher = Matcher()
    ...
    people = detections[detections.class_id == 0]
    hats =  detections[detections.class_id == 1]
    detections = people[matcher.match(people, hats)]
    ```
    """

    def __init__(
        self,
        max_missing_overlap: int = 60,
        max_missing_tracker: int = 30,
        min_overlap_threshold: float = 0.5,
        hysteresis: float = 0.4,
    ):
        """
        Initializes the Matcher instance with the given settings.

        Args:
            max_missing_overlap: Maximum number of frames an object can be missing before it is considered lost. Default is 60.
            max_missing_tracker: Maximum number of frames an object can be missing before it is removed from matching. Default is 30.
            min_overlap_threshold: Minimum overlap ratio required to consider two bounding boxes as overlapping. Default is 0.5.
            hysteresis: The hysteresis value used to determine the state of overlap detection. Default is 0.4.

        Example:
            ```python
            matcher = Matcher()
            ```
        """
        self.tracked_objects = []
        self.deleted_ids = deque()
        self.overlap_detector = OverlapDetector()

        # Create a MatcherSettings instance with the provided arguments
        self.settings = MatcherSettings(
            MAX_MISSING_OVERLAP=max_missing_overlap,
            MAX_MISSING_TRACKER=max_missing_tracker,
            MIN_OVERLAP_THRESHOLD=min_overlap_threshold,
            HYSTERESIS=hysteresis,
        )

    def match(self, base_object: Detections, *objects_to_match: Detections) -> List[bool]:
        """
        Checks to see if one lot of bbox Detections overlap with a List of other objects

        Args:
            base_object: The base detections to match other objects against, must be of type `Detections`, `Poses`, or `Segments` from `modlib.models.results` with bbox results.
            objects_to_match: Variadic list of detections to be checked against the base object, must be of type `Detections`, `Poses`, or `Segments` from `modlib.models.results` with bbox results.

        Returns:
            A mask of base detections that have matched overlap objects.

        Example:
            ```python
            matches = object1[matcher.match(object1, object2)]
            #This will give you the filtered object1 that have matched with object2
            ```
        """
        self.base_bboxes = base_object

        # Ensure tracked_objects has enough deques for the objects to match
        while len(self.tracked_objects) < len(objects_to_match):
            self.tracked_objects.append(deque())

        masks = []
        for i, obj in enumerate(objects_to_match):  # For each object to match
            self.tracked_object = self.tracked_objects[i]
            self.overlap_detector.filter(self.base_bboxes, obj)
            self.__update_tracked()
            mask = self.__get_mask()
            masks.append(mask)

        return_mask = masks[0]
        if len(masks) > 1:
            for m in range(len(masks) - 1):
                return_mask = [return_mask[i] and masks[m + 1][i] for i in range(len(return_mask))]

        return return_mask

    def __update_tracked(self):
        """
        Updates the matched Detection information over time. Also adds new matched Detections and removes them.
        """
        overlap_objects = self.overlap_detector
        self.deleted_ids = deque()
        overlap_objects_list = deque()
        for ool in overlap_objects:
            if ool[3] != -1:
                overlap_objects_list.append(DetectedObject(ool[0], ool[1], ool[2], ool[3], ool[4], self.settings))
        to_remove_index = []
        for to in self.tracked_object:
            if to in overlap_objects_list:
                index_of_match = overlap_objects_list.index(to)
                to.update(overlap_objects_list[index_of_match])
                del overlap_objects_list[index_of_match]
            else:
                if to.missing():
                    self.deleted_ids.append(to.tracker_id)
                    to_remove_index.append(self.tracked_object.index(to))

        for to_remove in to_remove_index:
            if to_remove < len(self.tracked_object):
                del self.tracked_object[to_remove]

        if len(overlap_objects_list):
            self.tracked_object.extend(overlap_objects_list)
        self.__merge_tensors()

    def __get_mask(self) -> List[bool]:
        """
        Generates output mask for object being matched

        Returns:
            The mask of detections that are matched for chosen object.
        """
        mask = np.ones(len(self.base_bboxes), dtype=bool)
        valid_objects = [o for o in self.tracked_object if o.uptime > 20]
        for i in range(len(mask)):
            found = False
            for to in valid_objects:
                if self.base_bboxes.tracker_id is not None and to.tracker_id is not None:
                    if to.tracker_id != self.base_bboxes.tracker_id[i]:
                        continue
                found = True
                if not to.get()[4]:  # Checks the Ratio of Intersect
                    mask[i] = False
            if found is False:
                mask[i] = False
        return mask

    def __merge_tensors(self):
        self.filtered_tracked_object = [to for to in self.tracked_object if not to.missing()]

        self.bbox = np.array([to.bbox for to in self.filtered_tracked_object])
        self.confidence = np.array([to.confidence for to in self.filtered_tracked_object])
        self.class_id = np.array([to.class_id for to in self.filtered_tracked_object])
        self.tracker_id = np.array([to.tracker_id for to in self.filtered_tracked_object])

    def __len__(self):
        return len(self.filtered_tracked_object)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, int, float, int]]:
        """
        Iterates over the filtered tracked objects.
        """
        for o in self.filtered_tracked_object:
            yield o.get()


class OverlapDetector:
    """
    The OverlapDetector class is used to identify overlapping objects.
    The objects to be detected for overlaps are specified during the construction of an OverlapDetector instance.
    This class can be used in scenarios where detecting spatial intersections or overlaps between objects is required.
    """

    def __init__(self):
        self.bboxir = np.empty((0,))
        self.maximum_bboxir = 0

    def filter(self, base_bboxes: np.ndarray, overlay_bboxes: np.ndarray) -> None:
        """
        Filters out the detections, both overlay and base, by calculating the intersect ratio.

        Args:
            base_bboxes: bboxes of a class_id that are being matched with all other objects
            overlay_bboxes: bboxes of a class_id that are to be matched with the base_bboxes
        """
        self.base_bboxes = base_bboxes.bbox
        self.overlay_bboxes = overlay_bboxes.bbox
        self.bbox = base_bboxes
        self.overlay = overlay_bboxes
        if len(self.base_bboxes) and len(self.overlay_bboxes):
            self.bboxir = self.bbox_intersect_ratio()
        elif len(self.base_bboxes):
            self.bboxir = np.zeros((len(self.base_bboxes), 1))
        else:
            return

    def bbox_intersect_ratio(self) -> np.ndarray:
        """
        Returns:
            The overlap ratio between the base bounding box and the overlaying bounding box.
        """
        # Expand dimensions to allow broadcasting
        b_areas = (self.overlay_bboxes[:, 2] - self.overlay_bboxes[:, 0]) * (
            self.overlay_bboxes[:, 3] - self.overlay_bboxes[:, 1]
        )

        # Broadcast the bounding boxes' coordinates for easier computation
        inter_x1 = np.maximum(self.base_bboxes[:, None, 0], self.overlay_bboxes[:, 0])
        inter_y1 = np.maximum(self.base_bboxes[:, None, 1], self.overlay_bboxes[:, 1])
        inter_x2 = np.minimum(self.base_bboxes[:, None, 2], self.overlay_bboxes[:, 2])
        inter_y2 = np.minimum(self.base_bboxes[:, None, 3], self.overlay_bboxes[:, 3])

        # Compute the area of intersection rectangle
        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

        overlap_ratio = inter_area / b_areas

        return overlap_ratio

    def __len__(self):
        if len(self.base_bboxes):
            self.maximum_bboxir = np.argmax(self.bboxir, axis=1)
            return len(self.bboxir)
        else:
            return 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, int, float]]:
        for i in range(len(self)):
            yield (
                self.bbox.bbox[i],
                self.bbox.confidence[i],
                self.bbox.class_id[i],
                self.bbox.tracker_id[i] if self.bbox.tracker_id is not None else None,
                self.bboxir[i, self.maximum_bboxir[i]],
            )