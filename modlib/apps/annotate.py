"""
MIT License

Copyright (c) 2022 Roboflow

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import math
import numpy as np

from modlib.apps.area import Area
from modlib.devices.frame import IMAGE_TYPE, Frame
from modlib.models import Detections, Poses, Segments, InstanceSegments

DEFAULT_COLOR_PALETTE = [
    "#FFCC14",
    "#FDBC4F",
    "#FAAC89",
    "#F89CC4",
    "#DB748F",
    "#BD4C59",
    "#A02424",
    "#BD3424",
    "#DB4525",
    "#F85525",
    "#F9713B",
    "#F98D52",
    "#FAA968",
    "#F9BA7F",
    "#F7CB95",
    "#F6DCAC",
    "#B9C6A5",
    "#7CB09F",
    "#3F9998",
    "#028391",
    "#02627B",
    "#014164",
    "#01204E",
    "#031C3D",
    "#06182D",
    "#08141C",
    "#20291F",
    "#383F21",
    "#505424",
    "#8A7C1F",
    "#C5A419",
]


def _validate_color_hex(color_hex: str):
    color_hex = color_hex.lstrip("#")
    if not all(c in "0123456789abcdefABCDEF" for c in color_hex):
        raise ValueError("Invalid characters in color hash")
    if len(color_hex) not in (3, 6):
        raise ValueError("Invalid length of color hash")


@dataclass
class Color:
    """
    Represents a color in RGB format. Used for specifying colors in annotations.
    """

    r: int  #: Red channel.
    g: int  #: Green channel.
    b: int  #: Blue channel.
    
    def __init__(self, r: int, g: int, b: int):
        """
        Initialize a color in RGB format. Used for specifying colors in annotations.
        Args:
            r: Red channel value (0-255).
            g: Green channel value (0-255).
            b: Blue channel value (0-255).

        Example usage:
            ```
            from modlib.apps.annotator import Annotator, Color
            frame.image = annotator.annotate_boxes(
                    frame=frame,
                    detections=matched_people,
                    labels=m_labels,
                    color=Color(0, 255, 0),
                    alpha = 0.2,
                )
            ```
        """
        if not all(isinstance(c, int) for c in (r, g, b)):
            raise ValueError("Color channels must be integers")
        if not all(0 <= c <= 255 for c in (r, g, b)):
            raise ValueError("Color channels must be between 0 and 255")
        self.r = r
        self.g = g
        self.b = b

    @classmethod
    def from_hex(cls, color_hex: str) -> Color:
        """
        Create a Color instance from a hex string.

        Args:
            color_hex: Hex string of the color.

        Returns:
            Instance representing the color.

        Example:
            ```
            >>> Color.from_hex('#ff00ff')
            Color(r=255, g=0, b=255)
            ```
        """
        _validate_color_hex(color_hex)
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)

    def as_rgb(self) -> Tuple[int, int, int]:
        """
        Returns the color as an RGB tuple.

        Returns:
            The RGB tuple.

        Example:
            ```
            >>> color.as_rgb()
            (255, 0, 255)
            ```
        """
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        """
        Returns the color as a BGR tuple.

        Returns:
            The BGR tuple.

        Example:
            ```
            >>> color.as_bgr()
            (255, 0, 255)
            ```
        """
        return self.b, self.g, self.r

    @classmethod
    def white(cls) -> Color:
        return Color.from_hex(color_hex="#ffffff")

    @classmethod
    def black(cls) -> Color:
        return Color.from_hex(color_hex="#000000")

    @classmethod
    def red(cls) -> Color:
        return Color.from_hex(color_hex="#ff0000")

    @classmethod
    def green(cls) -> Color:
        return Color.from_hex(color_hex="#00ff00")

    @classmethod
    def blue(cls) -> Color:
        return Color.from_hex(color_hex="#0000ff")

    @classmethod
    def yellow(cls) -> Color:
        return Color.from_hex(color_hex="#ffff00")

    def contrast_color(self) -> Color:
        """
        Returns a light or dark color for text based on the brightness of the color.

        Returns:
            A Color instance representing either black or white for better contrast.
        """
        # Calculate luminance
        luminance = 0.299 * self.r + 0.587 * self.g + 0.114 * self.b
        return Color.white() if luminance < 128 else Color.black()


@dataclass
class ColorPalette:
    colors: List[Color]  #: List of colors in the palette.

    @classmethod
    def default(cls) -> ColorPalette:
        """
        Returns a default color palette.

        Returns:
            A ColorPalette instance with default colors.

        Example:
            ```
            >>> ColorPalette.default()
            ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), ...])
            ```
        """
        return ColorPalette.from_hex(color_hex_list=DEFAULT_COLOR_PALETTE)

    @classmethod
    def from_hex(cls, color_hex_list: List[str]) -> ColorPalette:
        """
        Create a ColorPalette instance from a list of hex strings.

        Args:
            color_hex_list: List of color hex strings.

        Returns:
            A ColorPalette instance.

        Example:
            ```
            >>> ColorPalette.from_hex(['#ff0000', '#00ff00', '#0000ff'])
            ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), ...])
            ```
        """
        colors = [Color.from_hex(color_hex) for color_hex in color_hex_list]
        return cls(colors)

    def by_idx(self, idx: int) -> Color:
        """
        Return the color at a given index in the palette.

        Args:
            idx: Index of the color in the palette.

        Returns:
            The Color at the given index.

        Example:
            ```
            >>> color_palette.by_idx(1)
            Color(r=0, g=255, b=0)
            ```
        """
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        idx = idx % len(self.colors)
        return self.colors[int(idx)]


class Annotator:
    """
    Provides utility methods for annotating the frame with the provided image and corresponding detections.

    Example:
    ```
    from modlib.apps import Annotator

    annotator = Annotator()
    ...
    annotator.annotate_boxes(frame, detections, labels)
    ```
    """

    color: Union[Color, ColorPalette]  #: The color to draw the bounding box, can be a single color or a color palette.
    thickness: int  #: The thickness of the bounding box lines, default is 2.
    text_scale: float  #: The scale of the text on the bounding box, default is 0.5.
    text_thickness: int  #: The thickness of the text on the bounding box, default is 1.
    text_padding: int  #: The padding around the text on the bounding box, default is 10.

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(), 
        thickness: int = 2,
        text_scale: float = 0.5,
        text_thickness: int = 1, 
        text_padding: int = 10,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate_boxes(
        self,
        frame: Frame,
        detections: Detections,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
        color: Union[Color, ColorPalette] = None,
        alpha: float = None,
        corner_radius: int = 0,
        corner_length: int = 0,
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            frame: The frame to annotate, must be of type `Frame` from `modlib.devices`.
            detections: The detections to draw bounding boxes for, must be of type `Detections`, `Poses`, or `Segments` from `modlib.models.results`.
            labels: A list of labels for each detection. Defaults to `None`, in which case `class_id` is used.
            skip_label: Whether to skip drawing labels on the bounding boxes. Defaults to `False`.
            color: RGB color for bounding box edges and fill. Defaults to `None`.
            alpha: Transparency of the bounding box fill, between 0.0 and 1.0. Defaults to 0.5.
            corner_radius: Radius of the corners of the bounding boxes. Defaults to 0.
            corner_length: Length of the corners if `corner_radius` is 0. Defaults to 10.

        Returns:
            The annotated frame image.

        Example:
            ```python
            annotator.annotate_boxes(frame, detections, labels=["Person", "Car"], alpha=0.7)
            ```
        """
        if (
            not isinstance(detections, Detections)
            and not isinstance(detections, Poses)
            and not isinstance(detections, InstanceSegments)
        ):
            raise ValueError(
                "Input `detections` should be of type Detections, Poses, or InstanceSegments that contain bboxes"
            )

        # NOTE: Compensating for any introduced modified region of interest (ROI)
        # to ensure that detections are displayed correctly on top of the current `frame.image`.
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            detections.compensate_for_roi(frame.roi)

        h, w, _ = frame.image.shape
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.bbox[i]

            # Rescaling to frame size
            x1, y1, x2, y2 = (
                int(x1 * w),
                int(y1 * h),
                int(x2 * w),
                int(y2 * h),
            )

            if isinstance(detections, Detections) or isinstance(detections, InstanceSegments):
                class_id = detections.class_id[i] if detections.class_id is not None else None
                idx = class_id if class_id is not None else i
            else:  # Poses
                class_id, idx = "Person", 0

            if color is None:
                c = self.color.by_idx(idx) if isinstance(self.color, ColorPalette) else self.color
            else:
                c = color

            # Draw rectangle with possible rounded edges and infill
            self.rounded_rectangle(
                frame.image,
                x1,
                y1,
                x2,
                y2,
                c.as_bgr(),
                self.thickness,
                alpha,
                corner_radius,
                corner_length,
            )

            if skip_label:
                continue

            label = f"{class_id}" if (labels is None or len(detections) != len(labels)) else labels[i]
            self.set_label(
                image=frame.image,
                x=x1 - math.ceil(self.thickness / 2),
                y=y1 - math.ceil(self.thickness / 2),
                color=c.as_bgr(),
                label=label,
            )

        return frame.image

    def rounded_rectangle(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        thickness: int,
        alpha: float = None,
        corner_radius: int = 0,
        corner_length: int = 0,
    ):
        """
        Draws a rectangle with possible rounded edges and infill.

        Args:
            image: The input image as a NumPy array.
            x1: X-coordinate of the top-left corner of the rectangle.
            y1: Y-coordinate of the top-left corner of the rectangle.
            x2: X-coordinate of the bottom-right corner of the rectangle.
            y2: Y-coordinate of the bottom-right corner of the rectangle.
            color: BGR color for bounding box edges and fill.
            thickness: Thickness of the rectangle edges. Defaults to 2.
            alpha: Transparency of the rectangle fill, between 0.0 and 1.0. Defaults to 0.5.
            corner_radius: Radius of the rectangle corners. Defaults to 5.
            corner_length: Length of the rectangle corners if `corner_radius` is 0. Defaults to 10.
        """

        inner_pts = [
            (x1 + corner_radius, y1 + corner_radius),
            (x2 - corner_radius, y1 + corner_radius),
            (x2 - corner_radius, y2 - corner_radius),
            (x1 + corner_radius, y2 - corner_radius),
        ]

        if corner_length and not corner_radius:
            outer_pts = [
                # Top-left corner
                [(x1, y1), (x1 + corner_length, y1)],
                [(x1, y1), (x1, y1 + corner_length)],
                # Top-right corner
                [(x2, y1), (x2, y1 + corner_length)],
                [(x2, y1), (x2 - corner_length, y1)],
                # Bottom-right corner
                [(x2, y2), (x2 - corner_length, y2)],
                [(x2, y2), (x2, y2 - corner_length)],
                # Bottom-left corner
                [(x1, y2), (x1, y2 - corner_length)],
                [(x1, y2), (x1 + corner_length, y2)],
            ]
        else:
            outer_pts = [
                [(x1 + corner_radius, y1), (x2 - corner_radius, y1)],
                [(x2, y1 + corner_radius), (x2, y2 - corner_radius)],
                [(x2 - corner_radius, y2), (x1 + corner_radius, y2)],
                [(x1, y2 - corner_radius), (x1, y1 + corner_radius)],
            ]

        if alpha:
            # Create the cross corner points
            overlay = image.copy()
            corner_pts = np.array(
                [pt for i in range(4) for pt in (inner_pts[i], *outer_pts[i * len(outer_pts) // 4])], np.int32
            )
            cv2.fillPoly(overlay, [corner_pts], color=color)

            if corner_radius:
                for i, angle in enumerate([180, 270, 0, 90]):
                    cv2.ellipse(
                        overlay, inner_pts[i], (corner_radius, corner_radius), angle, 0, 90, color, thickness=cv2.FILLED
                    )

            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Draw the straight border lines
        for pt1, pt2 in outer_pts:
            cv2.line(image, pt1, pt2, color, thickness)

        # Draw the rounded corners
        if corner_radius:
            for i, angle in enumerate([180, 270, 0, 90]):
                cv2.ellipse(image, inner_pts[i], (corner_radius, corner_radius), angle, 0, 90, color, thickness)

    def annotate_area(
        self, frame: Frame, area: Area, color: Tuple[int, int, int], label: Optional[str] = None, alpha: float = None
    ) -> np.ndarray:
        """
        Draws a shape on the frame using the area containing points.

        Args:
            frame: The frame to annotate, must be of type `Frame` from `modlib.devices`.
            area: The area to draw, must be of type `Area` from `modlib.apps`.
            color: BGR color for the area.
            label: The text to display on the area.
            alpha: Transparency of the area fill, between 0.0 and 1.0. Defaults to 0.5.

        Returns:
            The annotated frame image.

        Example:
            ```python
            annotator.annotate_area(frame, area, color=(0, 255, 0), alpha=0.5)
            ```
        """
        h, w, _ = frame.image.shape
        resized_points = np.empty(area.points.shape, dtype=np.int32)
        resized_points[:, 0] = (area.points[:, 0] * w).astype(np.int32)
        resized_points[:, 1] = (area.points[:, 1] * h).astype(np.int32)
        resized_points = resized_points.reshape((-1, 1, 2))

        if alpha:
            overlay = frame.image.copy()
            cv2.fillPoly(frame.image, [resized_points], color=color)
            cv2.addWeighted(overlay, 1 - alpha, frame.image, alpha, 0, frame.image)

        # Draw the area on the image
        cv2.polylines(frame.image, [resized_points], isClosed=True, color=color, thickness=self.thickness)

        # Label
        if label:
            self.set_label(
                image=frame.image, x=resized_points[0][0][0], y=resized_points[0][0][1], color=color, label=label
            )

        return frame.image

    def set_label(self, image: np.ndarray, x: int, y: int, color: Tuple[int, int, int], label: str):
        """
        Draws text labels on the frame with background using the provided text and position.

        Args:
            image: The image to annotate, must be a NumPy array.
            x: X-coordinate for the label position.
            y: Y-coordinate for the label position.
            color: BGR color for the label background.
            label: The text to display.

        Example:
            ```python
            annotator.set_label(image, x=50, y=50, color=(255, 0, 0), label="Example Label")
            ```
        """

        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width, text_height = cv2.getTextSize(
            text=label,
            fontFace=font,
            fontScale=self.text_scale,
            thickness=self.text_thickness,
        )[0]

        text_x = x + self.text_padding
        text_y = y + self.text_padding + text_height

        text_background_x1 = x
        text_background_y1 = y

        text_background_x2 = x + 2 * self.text_padding + text_width
        text_background_y2 = y + 2 * self.text_padding + text_height

        # Draw background rectangle
        cv2.rectangle(
            img=image,
            pt1=(text_background_x1, text_background_y1),
            pt2=(text_background_x2, text_background_y2),
            color=color,
            thickness=cv2.FILLED,
        )

        # Draw text
        cv2.putText(
            img=image,
            text=label,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=self.text_scale,
            color=Color(*color[::-1]).contrast_color().as_bgr(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )

    def annotate_segments(self, frame: Frame, segments: Segments) -> np.ndarray:
        """
        Draws segmentation areas on the frame using the provided segments. 

        Args:
            frame: The frame to annotate, must be of type `Frame` from `modlib.devices`.
            segments: The segments defining the areas that will be drawn on the image, must be of type `Segments` from `modlib.models.results`.

        Returns:
            The annotated frame image.

        Example:
            ```python
            annotator.annotate_segments(frame, segments)
            ```
        """

        if not isinstance(segments, Segments):
            raise ValueError("Detections must be of type Segments.")

        # NOTE: Compensating for any introduced modified region of interest (ROI)
        # to ensure that detections are displayed correctly on top of the current `frame.image`.
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            segments.compensate_for_roi(frame.roi)

        h, w, _ = frame.image.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        for i in segments.indices:
            mask = segments.get_mask(i)
            c = self.color.by_idx(i) if isinstance(self.color, ColorPalette) else self.color
            colour = [(0, 0, 0, 0), (*c.as_bgr(), 255)]
            overlay_i = np.array(colour)[mask].astype(np.uint8)
            overlay += cv2.resize(overlay_i, (w, h))

        overlay[:, :, -1][overlay[:, :, -1] == 255] = 150
        frame.image = cv2.addWeighted(frame.image, 1, overlay[:, :, :3], 0.6, 0)

        return frame.image

    def annotate_oriented_boxes(
        self,
        frame: Frame,
        instance_segments: InstanceSegments,
        labels: Optional[List[str]] = None,
        skip_label: bool = False,
        color: Union[Color, ColorPalette] = None,
    ) -> np.ndarray:
        """
        Draws orientated bounding boxes on the frame using the detections provided.

        Args:
            frame: The frame to annotate, must be of type `Frame` from `modlib.devices`.
            instance_segments: The oriented bounding box data, must be of type `InstanceSegments` from `modlib.models.results`.
            labels: A list of labels for each detection. Defaults to `None`, in which case `class_id` is used.
            skip_label: Whether to skip drawing labels on the bounding boxes. Defaults to `False`.
            color: RGB color for bounding box edges and fill. Defaults to `None`.

        Returns:
            The annotated frame.image with orientated bounding boxes.
        """
        if not isinstance(instance_segments, InstanceSegments):
            raise ValueError("Instance segments must be of type InstanceSegments.")
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            instance_segments.compensate_for_roi(frame.roi)

        for i in range(len(instance_segments)):
            bbox = instance_segments.bbox[i]
            class_id = instance_segments.class_id[i] if instance_segments.class_id is not None else None
            idx = class_id if class_id is not None else i

            if color is None:
                c = self.color.by_idx(idx) if isinstance(self.color, ColorPalette) else self.color
            else:
                c = color

            # Rescaling to frame size
            bbox[:, 0] *= frame.width
            bbox[:, 1] *= frame.height
            bbox = bbox.astype(np.int32)

            cv2.drawContours(frame.image, [bbox], 0, c.as_bgr(), 2)

            if skip_label:
                continue
            label = f"{class_id}" if (labels is None or len(instance_segments) != len(labels)) else labels[i]
            self.set_label(
                image=frame.image,
                x=bbox[0][0] - math.ceil(self.thickness / 2),
                y=bbox[0][1] - math.ceil(self.thickness / 2),
                color=c.as_bgr(),
                label=label,
            )
        return frame.image

    def annotate_instance_segments(self, frame: Frame, instance_segments: InstanceSegments) -> np.ndarray:
        """
        Draws instance segmentation areas on the frame using the provided instance segments.

        Args:
            frame: The frame to annotate, must be of type `Frame` from `modlib.devices`.
            instance_segments: The instance segments defining the areas to draw, must be of type `InstanceSegments` from `modlib.models.results`.

        Returns:
            The annotated frame.image
        """

        if not isinstance(instance_segments, InstanceSegments):
            raise ValueError("Instance segments must be of type InstanceSegments.")
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            instance_segments.compensate_for_roi(frame.roi)

        h, w, _ = frame.image.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        for i in range(len(instance_segments)):
            mask = instance_segments.mask[i]
            c = self.color.by_idx(instance_segments.class_id[i]) if isinstance(self.color, ColorPalette) else self.color
            colour = [(0, 0, 0, 0), (*c.as_bgr(), 255)]
            overlay_i = np.array(colour)[mask].astype(np.uint8)
            overlay += cv2.resize(overlay_i, (w, h))

        overlay[:, :, -1][overlay[:, :, -1] == 255] = 150
        frame.image = cv2.addWeighted(frame.image, 1, overlay[:, :, :3], 0.6, 0)

        return frame.image

    def annotate_keypoints(
        self,
        frame: Frame,
        poses: Poses,
        num_keypoints: int = 17,
        skeleton: List[Tuple[int, int]] = [
            (5, 6),
            (11, 12),
            (5, 7),
            (7, 9),
            (5, 11),
            (11, 13),
            (13, 15),
            (6, 8),
            (8, 10),
            (6, 12),
            (12, 14),
            (14, 16),
        ],
        keypoint_radius: Optional[int] = 3,
        keypoint_color: Optional[Color] = Color.green(),
        line_color: Optional[Color] = Color.yellow(),
        keypoint_score_threshold: Optional[float] = 0.5,
    ) -> np.ndarray:
        """
        Draws the skeletons on the frame using the provided poses.

        Args:
            frame: The frame to annotate, must be of type `Frame` from `modlib.devices`.
            poses: The detections defining the skeletons that will be drawn on the image, must be of type `Poses` from `modlib.models.results`.
            num_keypoints: The number of unique keypoints in the poses object.
            skeleton: Edges between the keypoints that make up the skeleton to annotate. Defaults to `None`.
            keypoint_radius: The radius of the keypoints to be drawn. Defaults to 3.
            keypoint_color: The color of the keypoints. Defaults to green `(0, 255, 0)`.
            line_color: The color of the lines connecting keypoints. Defaults to yellow `(255, 255, 0)`.
            keypoint_score_threshold: The minimum score threshold for keypoints to be drawn. Keypoints with a score below this threshold will not be drawn. Defaults to 0.5.

        Returns:
            The annotated frame image.

        Example:
            ```python
            frame.image = annotator.annotate_keypoints(frame=frame, poses=poses)
            ```
        """

        if not isinstance(poses, Poses):
            raise ValueError("Detections must be of type Poses.")

        # NOTE: Compensating for any introduced modified region of interest (ROI)
        # to ensure that detections are displayed correctly on top of the current `frame.image`.
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            poses.compensate_for_roi(frame.roi)

        h, w, _ = frame.image.shape

        def draw_keypoints(poses, image, pose_idx, keypoint_idx, w, h, threshold=keypoint_score_threshold):
            if poses.keypoint_scores[pose_idx][keypoint_idx] >= threshold:
                x = int(poses.keypoints[pose_idx, keypoint_idx, 0] * w)
                y = int(poses.keypoints[pose_idx, keypoint_idx, 1] * h)
                cv2.circle(image, (x, y), keypoint_radius, keypoint_color.as_bgr(), -1)

        def draw_line(poses, image, pose_idx, keypoint1, keypoint2, w, h, threshold=keypoint_score_threshold):
            if (
                poses.keypoint_scores[pose_idx][keypoint1] >= threshold
                and poses.keypoint_scores[pose_idx][keypoint2] >= threshold
            ):
                x1 = int(poses.keypoints[pose_idx, keypoint1, 0] * w)
                y1 = int(poses.keypoints[pose_idx, keypoint1, 1] * h)
                x2 = int(poses.keypoints[pose_idx, keypoint2, 0] * w)
                y2 = int(poses.keypoints[pose_idx, keypoint2, 1] * h)
                cv2.line(image, (x1, y1), (x2, y2), line_color.as_bgr(), 2)

        for i in range(poses.n_detections):
            if poses.confidence[i] > keypoint_score_threshold:
                # Draw keypoints
                for j in range(num_keypoints):
                    draw_keypoints(poses, frame.image, i, j, w, h)

                # Draw skeleton lines
                for keypoint1, keypoint2 in skeleton:
                    draw_line(poses, frame.image, i, keypoint1, keypoint2, w, h)

        return frame.image

    def crop(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Crop a rectangular region from an image.

        Args:
            image: The input image as a NumPy array.
            x1: The x-coordinate of the top-left corner of the crop.
            y1: The y-coordinate of the top-left corner of the crop.
            x2: The x-coordinate of the bottom-right corner of the crop.
            y2: The y-coordinate of the bottom-right corner of the crop.

        Returns:
            The cropped region of the image.

        Example:
            ```python
            cropped_image = annotator.crop(frame.image, x1=50, y1=50, x2=200, y2=200)
            ```
        """
        return image[y1:y2, x1:x2]
