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

import math
from datetime import datetime
from typing import List, Tuple

import cv2
import numpy as np

from modlib.models import Detections
from modlib.devices.frame import Frame


class SpeedCalculator:
    """
    Calculates the speed of a moving objects and holds all tracked information.
    Uses bbox centers to calculate the change of distance over time
    For example, calculate the speed of object like:
    ```
    from modlib.apps.calculate import SpeedCalculator

    region = [(0.49, 0.0), (0.51, 0.0), (0.51, 1.0), (0.49, 1.0)]
    speed = SpeedCalculator(region)

    detections = tracker.update(frame, detections)
    speed.calculate(frame, detections)
    ```
    """

    region: List[Tuple[float, float]]  #: Points defining the polygon speed area e.g. [(x1, y1), (x2, y2), ...]

    def __init__(self, region: List[Tuple[float, float]] = [(0, 0), (1, 0), (1, 1), (0, 1)]):
        """
        Initializes the SpeedCalculator instance with the given region.

        Args:
            region: Points defining the polygon speed area. Default is [(0, 0), (1, 0), (1, 1), (0, 1)].

        Raises:
            ValueError: If the region is not defined by exactly 4 points or if any point is not a tuple of 2 values.

        Example:
            ```python
            region = [(0.49, 0.0), (0.51, 0.0), (0.51, 1.0), (0.49, 1.0)]
            speed_calculator = SpeedCalculator(region)
            ```
        """

        if len(region) != 4 or not all(len(point) == 2 for point in region):
            raise ValueError("""
                Region must be defined by exactly 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] and
                each point must be a tuple of 2 values.
            """)

        self.speed = {}
        self.tracked_id = []
        self.last_point = {}
        self.timestamp = {}
        self.track_history = {}
        self.stationary = {}
        self.current_speed = {}

        self.region = region

    def calculate(self, frame: Frame, detections: Detections):
        """
        Calculates the speed of detections by calculating the distance traveled over time using Detection's bboxes.
        While also storing tracked historic information to calculate the average speed in a given area.

        Args:
            frame: The current frame from camera
            detections: The set of Detections to check if the are in defined area.

        Example:
            ```python
            speed_calculator.calculate(frame, detections)
            ```
        """

        # Filter detections with invalid track id
        detections = detections[detections.tracker_id != -1]
        xc, yc = detections.center_points
        points = np.column_stack((xc * frame.width, yc * frame.height))

        for t, point in zip(detections.tracker_id, points):
            point = tuple(point)

            if t not in self.timestamp:
                self.timestamp[t] = frame.timestamp
                self.last_point[t] = point
                self.track_history[t] = []  # TODO: deque track history
                self.speed[t] = []
                self.stationary[t] = False
                self.current_speed[t] = 0.0

            self.track_history[t].append(point)

            if self.__intersect([self.last_point[t], self.track_history[t][-1]], self.region, frame.height, frame.width):
                moving = True
                totalx_movement = 0
                totaly_movement = 0
                counter = 0

                # in the last 10 points did x or y move forward 4 consecutive times?
                # if no -> stationary but can jitter, if yes: not stationary
                for i, point in reversed(
                    list(enumerate(self.track_history[t]))
                ):  # Loop to calculate if a detection is stationary
                    if counter > 10:
                        break
                    x_movement = self.track_history[t][i - 1][0] - self.track_history[t][i][0]
                    y_movement = self.track_history[t][i - 1][1] - self.track_history[t][i][1]

                    if x_movement > 0:
                        totalx_movement += 1
                    elif x_movement == 0:
                        totalx_movement += 0
                    else:
                        totalx_movement -= 1

                    if y_movement > 0:
                        totaly_movement += 1
                    elif y_movement == 0:
                        totaly_movement += 0
                    else:
                        totaly_movement -= 1
                    counter += 1

                if abs(totalx_movement) >= 4 or abs(totaly_movement) >= 4:
                    self.stationary[t] = False
                else:
                    self.stationary[t] = True

            else:
                moving = False

            if moving:
                if not self.stationary[t]:
                    # speed over last 2 frames
                    s = self.__speed_instance(
                        frame.timestamp, self.timestamp[t], self.track_history[t][-1], self.last_point[t]
                    )
                    if t not in self.tracked_id:
                        self.tracked_id.append(t)
                    self.speed[t].append(s)
                else:
                    s = 0.0
                self.current_speed[t] = s
            self.timestamp[t] = frame.timestamp
            self.last_point[t] = point

    def __speed_instance(self, timestamp1: str, timestamp2: str, p1: Tuple, p2: Tuple) -> float:
        """
        Calculates the speed of a object that has traveled between the 2 given points.

        Args:
            timestamp1: timestamp of the current frame
            timestamp2: timestamp of a previous frame
            p1: current point of the object
            p2: a previous point of the object

        Returns:
            Speed of the object
        """
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        time_diff = datetime.strptime(timestamp1, fmt) - datetime.strptime(timestamp2, fmt)
        time_diff = (time_diff.seconds * 1000) + (time_diff.microseconds / 1000000)
        if time_diff > 0:
            s = math.sqrt((abs((p2[0] - p1[0])) ** 2) + ((abs((p2[1] - p1[1]))) ** 2)) / time_diff
            return s

    def __check_segment(self, a: Tuple, b: Tuple, c: Tuple) -> bool:
        """
        Check to see if point is on segment on line
        """
        if (
            (b[0] <= max(a[0], c[0]))
            and (b[0] >= min(a[0], c[0]))
            and (b[1] <= max(a[1], c[1]))
            and (b[1] >= min(a[1], c[1]))
        ):
            return True
        return False

    def __orientation(self, a: Tuple, b: Tuple, c: Tuple) -> int:
        """
        Calculate the vector and its orientation in 2D space
        """
        v = (float(b[1] - a[1]) * (c[0] - b[0])) - (float(b[0] - a[0]) * (c[1] - b[1]))
        if v > 0:
            return 1
        elif v < 0:
            return 2
        else:
            return 0

    def __intersect(self, lineA: Tuple, lineB: Tuple, h: int, w: int) -> bool:
        """
        Calculate if a line intersects area

        Args:
            lineA: line of current object point and previous object point
            lineB: List of points that define the region
            h: height of frame
            w: width of frame

        Returns:
            If lineA intersects the area defined by lineB
        """
        lineB_scaled = np.array(
            (
                (lineB[0][0] * w, lineB[0][1] * h),
                (lineB[1][0] * w, lineB[1][1] * h),
                (lineB[2][0] * w, lineB[2][1] * h),
                (lineB[3][0] * w, lineB[3][1] * h),
            ),
            dtype=np.int32,
        )  # Assuming region of 4 points

        if lineA[0] == lineA[1]:
            return False

        p1check = cv2.pointPolygonTest(lineB_scaled, lineA[0], False)
        p2check = cv2.pointPolygonTest(lineB_scaled, lineA[1], False)
        if p1check >= 0 and p2check >= 0:
            return True

        for i in range(len(lineB)):
            b1 = [lineB[i][0], lineB[i][1]]
            b2 = [lineB[i - 1][0], lineB[i - 1][1]]
            i1 = self.__orientation(lineA[0], lineA[1], b1)
            i2 = self.__orientation(lineA[0], lineA[1], b2)
            i3 = self.__orientation(b1, b2, lineA[0])
            i4 = self.__orientation(b1, b2, lineA[1])

            if (i1 != i2) and (i3 != i4):  # Main intersect case
                return True
            # Special cases:
            if (i1 == 0) and self.__check_segment(lineA[0], lineA[1], b1):
                return True
            if (i2 == 0) and self.__check_segment(lineA[0], b2, b1):
                return True
            if (i3 == 0) and self.__check_segment(lineA[1], lineA[0], b1):
                return True
            if (i4 == 0) and self.__check_segment(lineA[1], b1, b2):
                return True
            return False

    def get_speed(self, t: int, average: bool = False) -> float | None:
        """
        Get speed by tracker ID.

        Args:
            t: Tracker ID to get speed for.
            average: Indicates whether to return the average or instantaneous speed. Default is False.

        Returns:
            Speed value in pixels per second, or None if the tracker ID is invalid (-1) or not found.

        Example:
            ```python
            speed = speed_calculator.get_speed(t, average=True)
            ```
        """
        if t == -1:
            return None
        elif t in self.tracked_id:
            if average:
                return sum(self.speed[t]) / len(self.speed[t])
            else:
                return self.current_speed[t]
        elif self.stationary[t]:
            return 0
        else:
            return None


def estimate_angle(k, focus_points, height, width):
    """
    Calculates the angle of the chosen keypoints.

    Args:
        k: Array of keypoints.
        focus_points: Indices of the keypoints to calculate the angle between.
        height: Height of the frame.
        width: Width of the frame.

    Returns:
        The calculated angle in degrees, or None if any keypoint is at (0.0, 0.0).

    Example:
        ```python
        angle = estimate_angle(keypoints, [0, 1, 2], frame.height, frame.width)
        ```
    """
    p1 = (int((k[focus_points[0]][0]) * width), int((k[focus_points[0]][1]) * height))
    p2 = (int((k[focus_points[1]][0]) * width), int((k[focus_points[1]][1]) * height))
    p3 = (int((k[focus_points[2]][0]) * width), int((k[focus_points[2]][1]) * height))
    if (0.0, 0.0) in [p1, p2, p3]:
        return
    p1, p2, p3 = np.array((p1)), np.array(p2), np.array(p3)
    a1 = (p1) - p2
    a2 = p3 - p2
    cos_a = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
    keypoint_angle = np.degrees(np.arccos(cos_a))
    if keypoint_angle > 180.0:
        keypoint_angle = 360 - keypoint_angle
    return round(keypoint_angle, 3)


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculates the distance between 2 given points.

    Args:
        point1: Tuple of x and y coordinates for a point
        point2: Tuple of x and y coordinates for a point

    Returns:
        Euclidean distance between the two points.

    Example:
        ```python
        distance = calculate_distance((0.0, 0.0), (1.0, 1.0))
        ```
    """
    return math.sqrt((abs((point2[0] - point1[0])) ** 2) + ((abs((point2[1] - point1[1]))) ** 2))


def calculate_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates a pairwise distance matrix between points defined by their x and y coordinates.

    Args:
        x: Array of x-coordinates for the points.
        y: Array of y-coordinates for the points.

    Returns:
        A square matrix where element [i,j] represents the Euclidean distance
                between point i and point j. The matrix is symmetric with zeros on the diagonal.

    Example:
        ```python
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        distances = calculate_distance_matrix(x, y)
        ```
    """
    p = np.column_stack((x, y))
    distance_matrix = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))
    return distance_matrix
