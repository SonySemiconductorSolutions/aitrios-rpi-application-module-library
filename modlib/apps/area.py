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

from typing import List, Tuple

import cv2
import numpy as np

from modlib.models import Detections


class Area:
    """
    Represents a polygonal area defined by a list of points.

    For example, declare a set of Areas could be done like this
    ```
    from modlib.apps import Area

    area_points = [[List of points of area],]
    areas = []
    for a in area_points:
        areas.append(Area(a))
    ```
    """

    points: List[Tuple[float, float]]  #: Points defining the polygon area e.g. [(x1, y1), (x2, y2), ...]

    def __init__(self, points: List[Tuple[float, float]]):
        for x, y in points:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(
                    f"Point ({x}, {y}) is out of bounds. Point coordinates must be defined relative between 0 and 1."
                )

        if len(points) < 3:
            raise ValueError("At least 3 points are required to form a polygon.")

        self.points = np.array(points, np.float32)

        # Check if the points form a valid polygon (no self-intersections)
        if not cv2.isContourConvex(self.points):
            raise ValueError("The points do not form a valid polygon (self-intersecting).")

    def contains(self, detections: Detections) -> List[bool]:
        """
        Checks to see if bbox Detections are in defined area

        Args:
            detections: The set of Detections to check if the are in defined area.

        Returns:
            The mask of detections that are in the current area.
        """
        mask = []
        for box in detections.bbox:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            point = (x_center, y_center)
            # Check if the point is inside the polygon
            result = cv2.pointPolygonTest(self.points, point, False)
            mask += [result >= 0]

        return mask
