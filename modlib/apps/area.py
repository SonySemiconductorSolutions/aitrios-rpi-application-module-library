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
import math
from modlib.models import Detections


class Area:
    """
    Represents a polygonal area defined by a list of points.
    Where one point (x, y) is the relative distance w.r.t. the width and height of the frame.

    For example, declare a set of Areas could be done like this
    ```
    from modlib.apps import Area

    area_points = [(0, 0), (1, 0), (1, 1),  (0, 1)] # Full area
    areas = []
    for a in area_points:
        areas.append(Area(a))
    ```
    """

    points: List[Tuple[float, float]]  #: Points defining the polygon area e.g. [(x1, y1), (x2, y2), ...]

    def __init__(self, points: List[Tuple[float, float]]):
        """
        Initializes an Area instance with the given polygon points.

        Args:
            points: A list of points defining the polygon, where each point is normalized to the range [0, 1] relative to the frame dimensions.
        Raises:
            ValueError: If any point is out of bounds, fewer than 3 points are provided, or the points form a self-intersecting polygon.
        """
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
        Checks whether the center of each detection's bounding box is inside the polygon.

        Args:
            detections: A set of detections containing bounding boxes, must be of type `Detections`, `Poses`, or `Segments` from `modlib.models.results`.

        Returns:
            A list of boolean values indicating whether each detection is inside the polygon.

        Example:
            ```python
            in_area = detections[area.contains(detections)]
            ```
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

    def anomaly_density(self, anomaly_mask: np.ndarray) -> float:
        """
        Calculate the proportion of anomaly pixels within this area's polygon.

        Args:
            anomaly_mask: A 2D array where non-zero values indicate anomalies.

        Returns:
            A value between 0 and 1 indicating the density of anomalies
            within the polygon. Returns 0.0 if the polygon has zero area.
        Raises:
            ValueError: If the anomaly mask is not a 3D array.

        Example:
            ```python
            density = area.anomaly_density(anomaly_mask)
            ```
        """
        if len(anomaly_mask.shape) != 3:
            raise ValueError("anomaly_mask must be a 3D array")
        anomaly_mask = anomaly_mask[:, :, 2]
        height, width = anomaly_mask.shape

        # Scale normalized points to pixel coordinates
        points = (self.points * [height, width]).astype(np.int32)

        # Create a binary mask for the polygon area
        mask = np.zeros_like(anomaly_mask, dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        # Count the number of pixels inside the polygon
        area_pixel_count = np.count_nonzero(mask)
        if area_pixel_count == 0:
            return 0.0  # Avoid division by zero

        # Apply the mask to the tensor (keep values where mask is 255, set the rest to 0)
        masked_tensor = cv2.bitwise_and(anomaly_mask, anomaly_mask, mask=mask)
        anomaly_point_count = np.count_nonzero(masked_tensor)

        return anomaly_point_count / area_pixel_count

    def to_dict(self) -> dict:
        """
        Converts the Area instance to a dictionary representation.

        Returns:
            A dictionary containing the type and points of the polygon.

        Example:
            ```python
            area_dict = area.to_dict()
            ```
        """
        points_list = self.points.tolist()
        return {"type": "Area", "points": points_list}

    @staticmethod
    def from_dict(data: dict) -> "Area":
        """
        Creates an Area instance from a dictionary representation.

        Args:
            data: A dictionary containing "points" key.

        Returns:
            An Area instance created from the provided "points" in the dictionary.
        """
        return Area(points=data["points"])


class Rectangle(Area):
    """
    A class representing a rectangle defined by two diagonal corners.
    """

    top_left: Tuple[float, float]  #: The (x, y) coordinates of the top-left corner of the rectangle.
    bottom_right: Tuple[float, float]  #: The (x, y) coordinates of the bottom-right corner of the rectangle.

    def __init__(self, top_left: Tuple[float, float], bottom_right: Tuple[float, float]):
        """
        Initializes a Rectangle instance with the given corner coordinates.
        The rectangle is assumed to be axis-aligned, meaning its sides are parallel to
        the x and y axes.

        Args:
            top_left: The (x, y) coordinates of the top-left corner.
            bottom_right: The (x, y) coordinates of the bottom-right corner.
        """

        self.top_left = top_left
        self.bottom_right = bottom_right

        x1, y1 = top_left
        x2, y2 = bottom_right
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        super().__init__(points)

    def to_dict(self) -> dict:
        """
        Serializes the Rectangle instance to a dictionary.

        Returns:
            A dictionary containing the rectangles's type, top left and bottom right corners.
        """
        return {
            "type": "Rectangle",
            "top_left": self.top_left,
            "bottom_right": self.bottom_right,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Rectangle":
        """
        Deserializes a Rectangle instance from a dictionary.

        Args:
            data: A dictionary containing 'top_left' and 'bottom_right' keys.

        Returns:
            A new Rectangle instance created from the provided data.
        """
        return cls(top_left=data["top_left"], bottom_right=data["bottom_right"])


class Circle(Area):
    """
    Represents a 2D circle shape, optionally defined by a center and either a radius or a point on the circumference.
    Automatically clamps the circle's radius to ensure it remains within the normalized bounds [0, 1].
    Inherits from the Area class.
    """

    center: Tuple[float, float]  #: The (x, y) coordinates of the center of the circle.
    point_circumference: Tuple[float, float]  #: The (x, y) coordinates of a point on the circumference.
    radius: float  #: The radius of the circle.
    aspect_ratio: float  #: The aspect ratio to adjust the circle's x-dimension.
    num_points: int  #: Number of points to use for approximating the circle's perimeter.

    def __init__(
        self,
        center: Tuple[float, float],
        point_circumference: Tuple[float, float] = None,
        radius: float = None,
        aspect_ratio: float = 1,
        num_points: int = 100,
    ):
        """
        Initializes a Circle instance with the given parameters.

        Args:
            center: The center point (x, y) of the circle.
            point_circumference: A point on the circle's circumference, used to calculate radius if radius is not provided.
            radius: The radius of the circle. If None, radius is calculated from point_circumference.
            aspect_ratio: The aspect ratio to adjust the circle's x-dimension.
            num_points: Number of points to use for approximating the circle's perimeter.
        """

        self.center = center
        if radius is not None:
            self.radius = radius
        else:
            self.radius = self.calculate_radius(point_circumference)

        self.radius = self.clamp_radius(self.center, self.radius)

        self.aspect_ratio = aspect_ratio

        cx, cy = center
        points = [
            (
                cx + self.radius * math.cos(2 * math.pi * i / num_points) / self.aspect_ratio,
                cy + self.radius * math.sin(2 * math.pi * i / num_points),
            )
            for i in range(num_points)
        ]

        super().__init__(points)

    def clamp_radius(self, center: Tuple[float, float], radius: float) -> float:
        """
        Ensures a circle defined by a center point (x, y) and radius stays within [0, 1] bounds.
        If it would go out of bounds, returns a clamped radius that keeps it inside.

        Args:
            center: The center point (x, y) of the circle.
            radius: The original radius of the circle.

        Returns:
            The adjusted radius to ensure the circle stays within [0, 1].
        """
        x, y = center

        # Maximum allowed radius to stay within bounds
        max_radius_x = min(x, 1 - x)
        max_radius_y = min(y, 1 - y)

        # The radius must be no larger than the smallest allowed in x or y
        max_radius = min(max_radius_x, max_radius_y)

        # Clamp radius
        clamped_radius = min(max(radius, 0), max_radius)

        return clamped_radius

    def to_dict(self) -> dict:
        """
        Serializes the Circle instance to a dictionary.

        Returns:
            A dictionary containing the circle's type, center, radius, and aspect ratio.
        """
        return {
            "type": "Circle",
            "center": self.center,
            "radius": self.radius,
            "aspect_ratio": self.aspect_ratio,
        }

    def calculate_radius(self, point_circumference: Tuple[float, float]) -> float:
        """
        Calculates the radius of the circle based on a point on the circumference.

        Args:
            point_circumference: A point (x, y) on the circumference of the circle.

        Returns:
            The calculated radius.
        """
        cx, cy = self.center
        x, y = point_circumference
        return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    @classmethod
    def from_dict(cls, data: dict) -> "Circle":
        """
        Deserializes a Circle instance from a dictionary.

        Args:
            data: A dictionary containing 'center', 'radius', and 'aspect_ratio' keys.

        Returns:
            A new Circle instance created from the provided data.
        """
        return cls(center=data["center"], radius=data["radius"], aspect_ratio=data["aspect_ratio"])
