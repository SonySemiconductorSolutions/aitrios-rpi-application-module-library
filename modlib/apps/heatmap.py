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

from typing import Callable

import cv2
import numpy as np

from modlib.devices.frame import Frame
from modlib.models import Detections


class Heatmap(object):
    """
    Functionality to create and visualize heatmaps based on object detection output data.

    Example:
    ```
    from modlib.apps import Heatmap

    heatmap = Heatmap(cell_size = 50)
    heatmap.update(frame, detections)
    ```
    """

    cell_size: int  #: Cell size that determines the heatmaps resolution

    def __init__(self, cell_size: int = 20) -> None:
        if not isinstance(cell_size, int):
            raise ValueError("Cell size must be of type 'int'")

        self.cell_size = cell_size

        # Initial frame width and height
        self.frame_size = None
        self.heatmap_size = None
        self.all_detections = Detections()

    def set_frame_size(self, width: int, height: int) -> None:
        """
        Change frame size.

        Args:
            width: Width of the frame size.
            height: Height of the frame size.
        """

        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("Both width and height must be of type 'int'")

        self.frame_size = (height, width)
        self.heatmap_size = (height // self.cell_size + 1, width // self.cell_size + 1)

    def set_cell_size(self, cell_size: int) -> None:
        """
        Change cell size.

        Args:
            cell_size: the heatmaps resolution
        """

        if not isinstance(cell_size, int):
            raise ValueError("cell_size must be of type 'int'")
        if self.frame_size is None:
            raise ValueError("Make sure to `set_frame_size` or call `update` once before changing the cell size.")

        self.cell_size = cell_size
        self.heatmap_size = (self.frame_size[0] // cell_size + 1, self.frame_size[1] // cell_size + 1)

    def update(self, frame: Frame, detections: Detections) -> np.ndarray:
        """
        Updates the heatmap of the objects on the given frame.

        Args:
            frame: The current frame to overlay the heatmap.
            detections: The object detections.

        Returns:
            The updated frame with the heatmap overlay.
        """
        if not isinstance(detections, Detections):
            raise ValueError("Input `detections` should be of type Detections")

        # Setting frame size
        if self.frame_size is None or frame.width != self.frame_size[1] or frame.height != self.frame_size[0]:
            self.set_frame_size(frame.width, frame.height)

        # Update detection window to create the heatmap for (storing all)
        self.all_detections += detections

        color_mapped_image = cv2.applyColorMap(self.create(), cv2.COLORMAP_JET)
        if frame.color_format == "RGB":
            color_mapped_image = cv2.cvtColor(color_mapped_image, cv2.COLOR_BGR2RGB)

        cv2.addWeighted(color_mapped_image, 0.4, frame.image, 1 - 0.4, 0, frame.image)

        return frame.image

    def create(self) -> np.ndarray:
        """Create the heatmap

        Raises:
            ValueError: If the input frames do not match the expected format.

        Returns:
            A heatmap matrix of shape `self.heatmap_size`, normalized to the range [0, 255].
        """

        # Create heatmap matrix
        heat_matrix = np.zeros(self.heatmap_size)

        for bbox, _, _, _ in self.all_detections:
            x1, y1, x2, y2 = (
                bbox[0] * self.frame_size[1],
                bbox[1] * self.frame_size[0],
                bbox[2] * self.frame_size[1],
                bbox[3] * self.frame_size[0],
            )

            r = ((y2 + y1) // 2) // self.cell_size
            c = ((x2 + x1) // 2) // self.cell_size

            heat_matrix[int(r), int(c)] += 1

        # Interpolate heatmap to target size & rescale
        heat_matrix = self.resize_image(
            heat_matrix, self.frame_size[0], self.frame_size[1], self.cell_size, self.bilinear_interpolation
        )
        m = np.max(heat_matrix)
        if m != 0:
            heat_matrix = heat_matrix / m
        heat_matrix = np.uint8(heat_matrix * 255)

        return heat_matrix

    @staticmethod
    def bilinear_interpolation(x: np.ndarray, y: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Perform bilinear interpolation for a given point (x, y) in the input image.

        Args:
            x: x-coordinate of the point to be interpolated.
            y: y-coordinate of the point to be interpolated.
            img: input image (2D array).

        Returns:
            Interpolated pixel values.
        """

        # Find the coordinates of the four pixels to be used for interpolation.
        x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
        x2, y2 = np.minimum(x1 + 1, img.shape[1] - 1), np.minimum(y1 + 1, img.shape[0] - 1)

        # Interpolate in the x-direction for the top and bottom pixel pairs.
        r1 = (x2 - x) * img[y1, x1] + (x - x1) * img[y1, x2]
        r2 = (x2 - x) * img[y2, x1] + (x - x1) * img[y2, x2]

        # Interpolate in the y-direction
        return (y2 - y) * r1 + (y - y1) * r2

    @staticmethod
    def resize_image(
        heat_matrix: np.ndarray,
        target_height: int,
        target_width: int,
        cell_size: int,
        interpolation_function: Callable[[float, float, np.ndarray], float],
    ) -> np.ndarray:
        """
        Resize the input heat_matrix to the target height and width using bilinear interpolation.

        Args:
            heat_matrix: input 2D array to be resized.
            target_height: desired height of the output array.
            target_width: desired width of the output array.
            cell_size: cell size of the heatmap.
            interpolation_function: function to be used for interpolation.

        Returns:
            The resized image.
        """

        # Calculate the scaling ratios in the x and y directions.
        src_height, src_width = heat_matrix.shape[:2]
        x_ratio = src_width / target_width
        y_ratio = src_height / target_height

        # Create a grid of indices representing the target image.
        row_indices, col_indices = np.ogrid[:target_height, :target_width]
        row_indices = (row_indices - 0.5 * cell_size) * y_ratio
        col_indices = (col_indices - 0.5 * cell_size) * x_ratio

        # Handle boundary cases to ensure indices don't go beyond valid coordinates
        row_indices = np.clip(row_indices, 0, src_height - 1)
        col_indices = np.clip(col_indices, 0, src_width - 1)

        # Interpolation for each point in the target grid according to scaling ratios
        interpolated_image = interpolation_function(col_indices, row_indices, heat_matrix)

        return interpolated_image
