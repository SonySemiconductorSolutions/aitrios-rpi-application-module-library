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
import numpy as np

from modlib.apps.area import Area
from modlib.devices.frame import IMAGE_TYPE, Frame
from modlib.models import Anomaly, Detections, Poses, Segments

DEFAULT_COLOR_PALETTE = [
    "#a351fb",
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#ff6347",
    "#4682b4",
    "#32cd32",
    "#ff69b4",
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
    Represents a color in RGB format.
    """

    r: int  #: Red channel.
    g: int  #: Green channel.
    b: int  #: Blue channel.

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


@dataclass
class ColorPalette:
    colors: List[Color]

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
    text_color: Color  #: The color of the text on the bounding box, default is black.
    text_scale: float  #: The scale of the text on the bounding box, default is 0.5.
    text_thickness: int  #: The thickness of the text on the bounding box, default is 1.
    text_padding: int  #: The padding around the text on the bounding box, default is 10.

    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette.default(),
        thickness: int = 2,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate_boxes(
        self, frame: Frame, detections: Detections, labels: Optional[List[str]] = None, skip_label: bool = False
    ) -> np.ndarray:
        """
        Draws bounding boxes on the frame using the detections provided.

        Args:
            frame: The frame to annotate.
            detections: The detections for which the bounding boxes will be drawn
            labels: An optional list of labels corresponding to each detection.
                If `labels` are not provided, corresponding `class_id` will be used as label.
            skip_label: Is set to `True`, skips bounding box label annotation.

        Returns:
            The annotated frame.image with bounding boxes.
        """
        if not isinstance(detections, Detections):
            raise ValueError("Detections must be of type Detections.")

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

            class_id = detections.class_id[i] if detections.class_id is not None else None
            idx = class_id if class_id is not None else i
            color = self.color.by_idx(idx) if isinstance(self.color, ColorPalette) else self.color
            cv2.rectangle(
                img=frame.image,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color.as_bgr(),
                thickness=self.thickness,
            )
            if skip_label:
                continue

            label = f"{class_id}" if (labels is None or len(detections) != len(labels)) else labels[i]

            self.set_label(image=frame.image, x=x1, y=y1, color=color.as_bgr(), label=label)

        return frame.image

    def annotate_area(
        self, frame: Frame, area: Area, color: Tuple[int, int, int], label: Optional[str] = None
    ) -> np.ndarray:
        """
        Draws a shape on the frame using the area containing points.

        Args:
            frame: The frame on which the area will be drawn. Must contain `frame.image`
            area: The area of a shape to draw on frame
            color: Tuple containing BGR value of the area

        Returns:
            The image with the area annotated.
        """
        h, w, _ = frame.image.shape
        resized_points = np.empty(area.points.shape, dtype=np.int32)
        resized_points[:, 0] = (area.points[:, 0] * w).astype(np.int32)
        resized_points[:, 1] = (area.points[:, 1] * h).astype(np.int32)
        resized_points = resized_points.reshape((-1, 1, 2))

        # Draw the area on the image
        cv2.polylines(frame.image, [resized_points], isClosed=True, color=color, thickness=2)

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
            image: The image to annotate from Frame.
            x: x coordinate for label position
            y: y coordinate for label position
            color: RGB value of background color
            label: text to be placed on frame
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
        text_y = y - self.text_padding

        text_background_x1 = x
        text_background_y1 = y - 2 * self.text_padding - text_height

        text_background_x2 = x + 2 * self.text_padding + text_width
        text_background_y2 = y

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
            color=self.text_color.as_rgb(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )

    def annotate_segments(self, frame: Frame, segments: Segments) -> np.ndarray:
        """
        Draws segmentation areas on the frame using the provided segments.

        Args:
            frame: The frame to annotate.
            segments: The segments defining the areas that will be drawn on the image.

        Returns:
            The annotated frame.image
        """

        if not isinstance(segments, Segments):
            raise ValueError("Detections must be of type Segments.")

        # NOTE: Compensating for any introduced modified region of interest (ROI)
        # to ensure that detections are displayed correctly on top of the current `frame.image`.
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            segments.compensate_for_roi(frame.roi)

        h, w, _ = frame.image.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        for i in segments.indeces:
            mask = segments.get_mask(i)
            c = self.color.by_idx(i) if isinstance(self.color, ColorPalette) else self.color
            colour = [(0, 0, 0, 0), (*c.as_bgr(), 255)]
            overlay_i = np.array(colour)[mask].astype(np.uint8)
            overlay += cv2.resize(overlay_i, (w, h))

        overlay[:, :, -1][overlay[:, :, -1] == 255] = 150
        frame.image = cv2.addWeighted(frame.image, 1, overlay[:, :, :3], 0.6, 0)

        return frame.image

    def annotate_poses(
        self,
        frame: Frame,
        poses: Poses,
        keypoint_radius: Optional[int] = 3,
        keypoint_color: Optional[Color] = Color.green(),
        line_color: Optional[Color] = Color.yellow(),
        keypoint_score_threshold: Optional[float] = 0.5,
    ) -> np.ndarray:
        """
        Draws skeletons on the frame using the provided poses.

        Args:
            frame: The frame to annotate.
            poses: The detections defining the skeletons that will be drawn on the image.
            keypoint_radius: The radius of the keypoints to be drawn. Defaults to 3.
            keypoint_color: The color of the keypoints. Defaults to green.
            line_color: The color of the lines connecting keypoints. Defaults to yellow.
            keypoint_score_threshold: The minimum score threshold for keypoints to be drawn.
                Keypoints with a score below this threshold will not be drawn. Defaults to 0.5.

        Returns:
            The annotated frame.image
        """

        if not isinstance(poses, Poses):
            raise ValueError("Detections must be of type Poses.")

        # NOTE: Compensating for any introduced modified region of interest (ROI)
        # to ensure that detections are displayed correctly on top of the current `frame.image`.
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            poses.compensate_for_roi(frame.roi)

        skeleton = [
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
        ]

        # keypoint_names = [
        #     'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder',
        #     'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
        #     'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
        # ]

        h, w, _ = frame.image.shape

        def draw_keypoints(poses, image, pose_idx, keypoint_idx, w, h, threshold=keypoint_score_threshold):
            if poses.keypoint_scores[pose_idx][keypoint_idx] >= threshold:
                y = int(poses.keypoints[pose_idx][2 * keypoint_idx] * h)
                x = int(poses.keypoints[pose_idx][2 * keypoint_idx + 1] * w)
                cv2.circle(image, (x, y), keypoint_radius, keypoint_color.as_bgr(), -1)

        def draw_line(poses, image, pose_idx, keypoint1, keypoint2, w, h, threshold=keypoint_score_threshold):
            if (
                poses.keypoint_scores[pose_idx][keypoint1] >= threshold
                and poses.keypoint_scores[pose_idx][keypoint2] >= threshold
            ):
                y1 = int(poses.keypoints[pose_idx][2 * keypoint1] * h)
                x1 = int(poses.keypoints[pose_idx][2 * keypoint1 + 1] * w)
                y2 = int(poses.keypoints[pose_idx][2 * keypoint2] * h)
                x2 = int(poses.keypoints[pose_idx][2 * keypoint2 + 1] * w)
                cv2.line(image, (x1, y1), (x2, y2), line_color.as_bgr(), 2)

        for i in range(poses.n_detections):
            if poses.confidence[i] > keypoint_score_threshold:
                # Draw keypoints
                for j in range(17):
                    draw_keypoints(poses, frame.image, i, j, w, h)

                # Draw skeleton lines
                for keypoint1, keypoint2 in skeleton:
                    draw_line(poses, frame.image, i, keypoint1, keypoint2, w, h)

        return frame.image
