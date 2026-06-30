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
from modlib.models import Detections, Poses, Segments, InstanceSegments, OBB

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
            The annotated frame.image

        Example:
            ```python
            annotator.annotate_boxes(frame, detections, labels=["Person", "Car"], alpha=0.7)
            ```
        """
        if (
            not isinstance(detections, Detections)
            and not isinstance(detections, Poses)
            and not isinstance(detections, InstanceSegments)
            and not isinstance(detections, OBB)
        ):
            raise ValueError(
                "Input `detections` should be of type Detections, Poses, InstanceSegments, or OBB that contain bboxes"
            )

        # scale ALL bboxes at once to frame size and account for ROI if needed
        detections = scale_bboxes(detections, frame.image.shape, frame.image_type, frame.roi)

        h, w, _ = frame.image.shape
        for i in range(len(detections)):
            if isinstance(detections, OBB):
                x, y, w, h = detections.bbox[i]
                a = detections.angle[i]
            else:
                x1, y1, x2, y2 = detections.bbox[i]
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                a = 0.0

            if isinstance(detections, Detections) or isinstance(detections, InstanceSegments) or isinstance(detections, OBB):
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
                x=x,
                y=y,
                w=w,
                h=h,
                a=a,
                color=c.as_bgr(),
                thickness=self.thickness,
                alpha=alpha,
                corner_radius=corner_radius,
                corner_length=corner_length,
            )

            if skip_label:
                continue

            label = f"{class_id}" if (labels is None or len(detections) != len(labels)) else labels[i]
            self.set_label(
                image=frame.image,
                x=x - w/2 - math.ceil(self.thickness / 2),
                y=y - h/2 - math.ceil(self.thickness / 2),
                color=c.as_bgr(),
                label=label,
            )

        return frame.image

    @staticmethod
    def _rotate_local_point(px: float, py: float, x: float, y: float, a: float) -> Tuple[int, int]:
        # Local points use image coordinates where y+ means down; this matches the rotation math in OBB.
        x_rot = px * math.cos(a) - py * math.sin(a) + x
        y_rot = px * math.sin(a) + py * math.cos(a) + y
        return int(round(x_rot)), int(round(y_rot))

    @staticmethod
    def _arc_points(
        cx: float,
        cy: float,
        r: float,
        start_theta: float,
        end_theta: float,
        num: int,
    ) -> List[Tuple[float, float]]:
        thetas = np.linspace(start_theta, end_theta, num=num, endpoint=True)
        return [(cx + r * float(math.cos(t)), cy + r * float(math.sin(t))) for t in thetas]

    def rounded_rectangle(
        self,
        image: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
        a: float,
        color: Tuple[int, int, int],
        thickness: int,
        alpha: float = None,
        corner_radius: int = 0,
        corner_length: int = 0,
    ) -> np.ndarray:
        """
        Draws a (potentially rotated) rectangle with possible rounded edges and infill.

        Args:
            image: The input image as a NumPy array.
            x: X-coordinate of the rectangle center.
            y: Y-coordinate of the rectangle center.
            w: Width of the rectangle (center-based).
            h: Height of the rectangle (center-based).
            a: Rotation angle in radians. Positive values rotate clockwise in image coordinates.
            color: BGR color for bounding box edges and fill.
            thickness: Thickness of the rectangle edges. Defaults to 2.
            alpha: Transparency of the rectangle fill, between 0.0 and 1.0. Defaults to 0.5.
            corner_radius: Radius of the rectangle corners. Defaults to 5.
            corner_length: Length of the rectangle corners if `corner_radius` is 0. Defaults to 10.

        Returns:
            The annotated image.
        """
        half_w = w / 2.0
        half_h = h / 2.0
        left, right = -half_w, half_w
        top, bottom = -half_h, half_h

        # Clamp radii/length so they remain meaningful for the given rectangle size.
        max_radius = int(max(0, min(half_w, half_h)))
        corner_radius = max(0, min(corner_radius, max_radius))
        if corner_length and not corner_radius:
            corner_length = max(0, min(corner_length, int(min(half_w, half_h))))

        # Corner-marks mode: only draw L-shaped corners (when corner_radius == 0).
        if corner_length and not corner_radius:
            inner_pts = [(left, top), (right, top), (right, bottom), (left, bottom)]
            outer_pts = [
                # Top-left corner
                [(left, top), (left + corner_length, top)],
                [(left, top), (left, top + corner_length)],
                # Top-right corner
                [(right, top), (right, top + corner_length)],
                [(right, top), (right - corner_length, top)],
                # Bottom-right corner
                [(right, bottom), (right - corner_length, bottom)],
                [(right, bottom), (right, bottom - corner_length)],
                # Bottom-left corner
                [(left, bottom), (left, bottom - corner_length)],
                [(left, bottom), (left + corner_length, bottom)],
            ]

            if alpha is not None and alpha > 0:
                overlay = image.copy()
                corner_pts_local = np.array(
                    [pt for i in range(4) for pt in (inner_pts[i], *outer_pts[i * len(outer_pts) // 4])], dtype=np.float32
                )
                corner_pts = np.array([self._rotate_local_point(px, py, x, y, a) for px, py in corner_pts_local], dtype=np.int32)
                cv2.fillPoly(overlay, [corner_pts], color=color)
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            for (pt1x, pt1y), (pt2x, pt2y) in outer_pts:
                cv2.line(image, self._rotate_local_point(pt1x, pt1y, x, y, a), self._rotate_local_point(pt2x, pt2y, x, y, a), color, thickness)
            return image

        # Build the rounded-rectangle outline in local coordinates (unrotated).
        if corner_radius > 0:
            cr = float(corner_radius)
            arc_steps = max(8, int(math.ceil(cr)))

            # Clockwise order starting at top edge after the top-left arc.
            outline_local: List[Tuple[float, float]] = []
            outline_local.append((left + cr, top))  # P0
            outline_local.append((right - cr, top))  # P1

            # Top-right arc: (-pi/2 -> 0)
            center_tr = (right - cr, top + cr)
            outline_local.extend(self._arc_points(center_tr[0], center_tr[1], cr, -math.pi / 2, 0, arc_steps)[1:])
            outline_local.append((right, bottom - cr))  # P3

            # Bottom-right arc: (0 -> pi/2)
            center_br = (right - cr, bottom - cr)
            outline_local.extend(self._arc_points(center_br[0], center_br[1], cr, 0, math.pi / 2, arc_steps)[1:])
            outline_local.append((left + cr, bottom))  # P5

            # Bottom-left arc: (pi/2 -> pi)
            center_bl = (left + cr, bottom - cr)
            outline_local.extend(self._arc_points(center_bl[0], center_bl[1], cr, math.pi / 2, math.pi, arc_steps)[1:])
            outline_local.append((left, top + cr))  # P7

            # Top-left arc: (pi -> 3pi/2) => ends back at P0
            center_tl = (left + cr, top + cr)
            outline_local.extend(self._arc_points(center_tl[0], center_tl[1], cr, math.pi, 3 * math.pi / 2, arc_steps)[1:])
        else:
            # Sharp rectangle outline
            outline_local = [(left, top), (right, top), (right, bottom), (left, bottom)]

        outline_pts = np.array([self._rotate_local_point(px, py, x, y, a) for px, py in outline_local], dtype=np.int32)

        if alpha is not None and alpha > 0:
            overlay = image.copy()
            cv2.fillPoly(overlay, [outline_pts], color=color)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.polylines(image, [outline_pts], isClosed=True, color=color, thickness=thickness)

        return image

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
            The annotated frame.image

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

    def set_label(self, image: np.ndarray, x: int, y: int, color: Tuple[int, int, int], label: str) -> np.ndarray:
        """
        Draws text labels on the frame with background using the provided text and position.

        Args:
            image: The image to annotate, must be a NumPy array.
            x: X-coordinate for the label position.
            y: Y-coordinate for the label position.
            color: BGR color for the label background.
            label: The text to display.

        Returns:
            The annotated image.

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
            pt1=(int(text_background_x1), int(text_background_y1)),
            pt2=(int(text_background_x2), int(text_background_y2)),
            color=color,
            thickness=cv2.FILLED,
        )

        # Draw text
        cv2.putText(
            img=image,
            text=label,
            org=(int(text_x), int(text_y)),
            fontFace=font,
            fontScale=self.text_scale,
            color=Color(*color[::-1]).contrast_color().as_bgr(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )

        return image

    def annotate_segments(self, frame: Frame, segments: Segments) -> np.ndarray:
        """
        Draws segmentation areas on the frame using the provided segments.

        Args:
            frame: The frame to annotate, must be of type `Frame` from `modlib.devices`.
            segments: The segments defining the areas that will be drawn on the image, must be of type `Segments` from `modlib.models.results`.

        Returns:
            The annotated frame.image

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

    def annotate_instance_segments(self, frame: Frame, instance_segments: InstanceSegments) -> np.ndarray:
        """
        Draws instance segmentation areas on the frame using the provided instance segments.

        Args:
            frame: The frame to annotate, must be of type `Frame` from `modlib.devices`.
            instance_segments: The instance segments defining the areas to draw, must be of type `InstanceSegments` from `modlib.models.results`.

        Returns:
            The annotated frame.image

        Example:
            ```python
            annotator.annotate_instance_segments(frame, instance_segments)
            ```
        """

        if not isinstance(instance_segments, InstanceSegments):
            raise ValueError("Instance segments must be of type InstanceSegments.")
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            instance_segments.compensate_for_roi(frame.roi)

        h, w, _ = frame.image.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        mask_h, mask_w = instance_segments.mask_shape
        bbox = instance_segments.bbox
        mask_crops = instance_segments.mask_crops

        for i in range(len(mask_crops)):
            yi1, yi2, xi1, xi2 = InstanceSegments._bbox_to_slices(bbox[i], mask_h, mask_w)
            crop = np.asarray(mask_crops[i], dtype=np.uint8)
            ch, cw = yi2 - yi1, xi2 - xi1
            if ch <= 0 or cw <= 0 or crop.shape != (ch, cw):
                continue
            c = self.color.by_idx(instance_segments.class_id[i]) if isinstance(self.color, ColorPalette) else self.color
            colour = [(0, 0, 0, 0), (*c.as_bgr(), 255)]
            overlay_i = np.zeros((mask_h, mask_w, 4), dtype=np.uint8)
            overlay_i[yi1:yi2, xi1:xi2] = np.array(colour)[crop].astype(np.uint8)
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
            The annotated frame.image

        Example:
            ```python
            annotator.annotate_keypoints(frame=frame, poses=poses)
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



def _scale_xyxy(
    dets: Union[Detections, Poses, InstanceSegments],
    h: int,
    w: int,
) -> Union[Detections, Poses, InstanceSegments]:
    # Scale bboxes to frame size
    scaled_bboxes = np.zeros_like(dets.bbox, dtype=int)
    scaled_bboxes[:, 0] = (dets.bbox[:, 0] * w).astype(int)
    scaled_bboxes[:, 1] = (dets.bbox[:, 1] * h).astype(int)
    scaled_bboxes[:, 2] = (dets.bbox[:, 2] * w).astype(int)
    scaled_bboxes[:, 3] = (dets.bbox[:, 3] * h).astype(int)
    dets.bbox = scaled_bboxes
    return dets


def _scale_xywha(
    dets: OBB,
    h: int,
    w: int,
) -> OBB:
    # OBB stores (cx, cy, bw, bh, angle). Under anisotropic scaling (sx != sy),
    # side lengths and orientation must be transformed geometrically.
    scaled_bboxes = np.zeros_like(dets.bbox, dtype=int)
    scaled_bboxes[:, 0] = (dets.bbox[:, 0] * w).astype(int)
    scaled_bboxes[:, 1] = (dets.bbox[:, 1] * h).astype(int)

    # u: Width-axis unit vector u=(cos(a), sin(a)) after anisotropic scaling.
    # v: Height-axis unit vector v=(-sin(a), cos(a)) after anisotropic scaling.
    ux = np.cos(dets.angle) * w
    uy = np.sin(dets.angle) * h
    vx = -np.sin(dets.angle) * w
    vy = np.cos(dets.angle) * h

    scaled_bboxes[:, 2] = (dets.bbox[:, 2] * np.sqrt(ux**2 + uy**2)).astype(int)
    scaled_bboxes[:, 3] = (dets.bbox[:, 3] * np.sqrt(vx**2 + vy**2)).astype(int)
    dets.bbox = scaled_bboxes
    dets.angle = np.arctan2(uy, ux).astype(np.float32)
    return dets


def scale_bboxes(
    detections: Union[Detections, Poses, InstanceSegments, OBB],
    image_shape: Tuple[int, int, int],
    image_type: IMAGE_TYPE,
    roi: Tuple[float, float, float, float],
) -> Union[Detections, Poses, InstanceSegments, OBB]:
    """
    Scale multiple (oriented) bounding boxes from normalized coordinates to image frame size and return updated detections.
    If the input image is not an INPUT_TENSOR, compensates for any Region of Interest (ROI) applied during pre-processing,
    so that bounding boxes are correctly mapped back to the current frame.
    Typically, this happens for models that preserve aspect ratio.

    Args:
        detections: Object with bounding boxes in normalized coordinates.
        image_shape: Tuple (height, width, _) representing the image shape.
        image_type: IMAGE_TYPE enum specifying the input image type.
        roi: ROI tuple specifying the region of interest (x, y, w, h).

    Returns:
        Object with bounding boxes (and possibly angles) scaled to frame pixel coordinates.
    """

    # Copy detections so as not to mutate the input
    dets = detections.copy()

    # Take into account any ROI, also handles preserve aspect ratio in pre-processing
    # as implemented by IMX500 ISP (crops and resized image to input tensor size).
    h, w, _ = image_shape
    if image_type != IMAGE_TYPE.INPUT_TENSOR:
        # image_ratio = w / h
        dets.compensate_for_roi(roi)

    if (
        isinstance(detections, Detections)
        or isinstance(detections, Poses)
        or isinstance(detections, InstanceSegments)
    ):
        return _scale_xyxy(dets, h, w)
    elif isinstance(detections, OBB):
        return _scale_xywha(dets, h, w)
    else:
        raise ValueError(
            "Input `detections` should be of type Detections, Poses, InstanceSegments, or OBB that contain bboxes"
        )
