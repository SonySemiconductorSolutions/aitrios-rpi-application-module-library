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

import numpy as np

from modlib.models import Detections, Poses


class ObjectCounter:
    """
    A class responsible for keeping track of all classes detected and the amount of them over time. Can be used with or
    without a tracker_id and can be send subsets of Detections and Poses to count certain conditions.
    """

    def __init__(self):
        """
        Initializes the ObjectCounter instance.

        Attributes:
            counters (dict): A dictionary to store counts for each detected class.
            valid_IDs (np.ndarray): An array of valid tracker IDs.
            uptime (dict): A dictionary to track the uptime of each tracker ID.

        Example:
            ```python
            object_counter = ObjectCounter()
            ```
        """

        self.counters = {}
        self.valid_IDs = np.array([])
        self.uptime = {}

    def update(self, detections: Detections) -> None:
        """
        Updates counters for all classes detected in the Detections object. Increments counts for already detected classes.

        Args:
            detections: The detections to process and count. If tracker_id is used, it will only count an ID once it has been seen for more than 20 frames.

        Example:
            ```python
            object_counter.update(detections)
            ```
        """
        if detections.tracker_id is not None:
            for _, s, c, t in detections:
                if t not in self.valid_IDs:
                    if str(t) in self.uptime:
                        if self.uptime[str(t)] > 20:
                            self.valid_IDs = np.append(self.valid_IDs, t)
                            if str(c) in self.counters:
                                self.counters[str(c)] += 1
                            else:
                                self.counters[str(c)] = 1
                            del self.uptime[str(t)]
                        else:
                            self.uptime[str(t)] += 1
                    else:
                        self.uptime[str(t)] = 1
        else:
            for _, s, c, t in detections:
                if str(c) in self.counters:
                    self.counters[str(c)] += 1
                else:
                    self.counters[str(c)] = 1

    def update_pose(self, poses: Poses) -> None:
        """
        Updates counters for pose detections. Since Poses do not have `class_id`, all detections are counted under the class ID `1`.

        Args:
            poses: The pose detections to process and count. If tracker is used, it will only count an ID once it has been seen for more than 30 frames.

        Example:
            ```python
            object_counter.update_pose(poses)
            ```
        """
        if poses.tracker is not None:
            for k, s, _, b, t in poses:
                if t == -1:
                    continue
                if t not in self.valid_IDs:
                    if str(t) in self.uptime:
                        if self.uptime[str(t)] > 30:
                            self.valid_IDs = np.append(self.valid_IDs, t)
                            if str("1") in self.counters:
                                self.counters["1"] += 1
                            else:
                                self.counters["1"] = 1
                            del self.uptime[str(t)]
                        else:
                            self.uptime[str(t)] += 1
                    else:
                        self.uptime[str(t)] = 1
        else:
            for k, s, _, b, t in poses:
                if str(1) in self.counters:
                    self.counters["1"] += 1
                else:
                    self.counters["1"] = 1

    def get(self, class_id: int) -> int:
        """
        Gets the value of the counter by it's ID

        Args:
            class_id: The ID of the class user wants value of

        Returns:
            The count for the specified class ID. Returns 0 if the class ID is not found.

        Example:
            ```python
            count = object_counter.get(class_id=1)
        """
        if str(class_id) in self.counters:
            return self.counters[str(class_id)]
        return 0
