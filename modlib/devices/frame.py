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
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from ..models import COLOR_FORMAT, ROI, Anomaly, Classifications, Detections, Poses, Segments, InstanceSegments


CV2_WINDOWS = set()


@dataclass
class IMAGE_TYPE:
    """
    Representation of the available image types provided in `frame.image`. Can be used as e.g. `IMAGE_TYPE.VGA`
    """

    VGA = "VGA"
    INPUT_TENSOR = "input_tensor"
    SOURCE = "source"


@dataclass
class RESULT_TYPE:
    """
    Utility dataclass for collecting all Result types.
    """

    Classifications = Classifications
    Detections = Detections
    Poses = Poses
    Segments = Segments
    InstanceSegments = InstanceSegments
    Anomaly = Anomaly


class Frame:
    """
    Represents a frame in a device stream.

    ```
    with device as stream:
        for frame in stream:
            ...
    ```

    The frame object has the following available elements:
    """

    timestamp: str  #: The timestamp of the frame.
    image: np.ndarray  #: The image data of the frame.
    image_type: IMAGE_TYPE  #: Specification of what is visualised in `frame.image`.
    width: int  #: The width of the frame.
    height: int  #: The height of the frame.
    channels: int  #: The number of channels in the frame.
    detections: Union[Classifications, Detections, Poses, Segments, InstanceSegments, Anomaly]  #: The detections in the frame.
    new_detection: bool  #: Flag if the provided detections are updated or an old copy.
    fps: float  #: The frames per second of the video stream.
    dps: float  #: The detections per second in the video stream.
    color_format: COLOR_FORMAT  #: The color format of the frame. Defaults to `RGB`.
    input_tensor: Optional[np.ndarray]  #: The input tensor of the frame. Defaults to None.
    roi: Optional[ROI]  #: Relative ROI of the input tensor. Defaults to None.

    def __init__(
        self,
        timestamp: str,
        image: np.ndarray,
        image_type: IMAGE_TYPE,
        width: int,
        height: int,
        channels: int,
        detections: Union[Classifications, Detections, Poses, Segments, InstanceSegments, Anomaly],
        new_detection: bool,
        fps: float,
        dps: float,
        color_format: COLOR_FORMAT = COLOR_FORMAT.RGB,
        input_tensor: Optional[np.ndarray] = None,
        roi: Optional[ROI] = None,
    ):
        """
        Initialize an Frame object.
        """
        self.timestamp = timestamp
        self._image = image
        self.image_type = image_type
        self.width = width
        self.height = height
        self.channels = channels
        self._detections = detections
        self.new_detection = new_detection
        self.fps = fps
        self.dps = dps
        self.color_format = color_format
        self._input_tensor = input_tensor
        self.roi = roi

    @property
    def image(self) -> np.ndarray:
        """
        Get the image data of the frame.

        Returns:
            The image data of the frame.

        Raises:
            ValueError: When running headless and the image is not available.
        """
        if self._image is not None:
            return self._image
        else:
            raise ValueError("Running headless: `frame.image` unavailable.\n")

    @image.setter
    def image(self, value: np.ndarray):
        self._image = value

    @property
    def detections(self) -> Union[Classifications, Detections, Poses, Segments, InstanceSegments, Anomaly]:
        """
        Get the detections in the frame.

        Returns:
            The detections in the frame.

        Raises:
            ValueError: If no model is running.
        """
        if self._detections is not None:
            return self._detections
        else:
            raise ValueError("No model is running: `frame.detections` unavailable.\n")

    @detections.setter
    def detections(self, value: Union[Classifications, Detections, Poses, Segments, InstanceSegments, Anomaly]):
        self._detections = value

    @property
    def input_tensor(self) -> np.ndarray:
        """
        Get the input tensor of the frame.

        Returns:
            The input tensor of the frame.
        """
        if self._input_tensor is not None:
            # return self._input_tensor
            raise NotImplementedError("TODO get the real ISP output (for framework input)")
        else:
            raise ValueError(
                """
                Input tensor not enabled: `frame.input_tensor` unavailable.
                Initialize device with `enable_input_tensor=True`
            """
            )

    @input_tensor.setter
    def input_tensor(self, value: np.ndarray):
        self._input_tensor = value

    def display(
        self,
        show_fps_dps: bool = True,
        cropping: Optional[Union[ROI, Tuple[float, float, float, float]]] = None,
        rotate: Optional[int] = None,
        flip: Optional[int] = None,
        resize_image: bool = False,
        window_name: str = "Application",
    ):
        """
        Display the frame with various options for visualization.

        Args:
            show_fps_dps: If True, display the frames per second (FPS) and detections per second (DPS) on the image.
            cropping: The region of interest to display, either as a named tuple ROI or a tuple of four floats
                (left, top, width, height). Defaults to None for no cropping.
            rotate: The rotation code for the image. Use cv2 rotation codes:
                > 0 or cv2.ROTATE_90_CLOCKWISE: Rotate the image 90 degrees clockwise.
                > 1 or cv2.ROTATE_180: Rotate the image 180 degrees.
                > 2 or cv2.ROTATE_90_COUNTERCLOCKWISE: Rotate the image 90 degrees counterclockwise.
                > None: No rotation.
            flip: The flip code for the image. Use 0 for vertical, 1 for horizontal, -1 for both, or None for no flip.
            resize_image: If True, resize the image to fit the display window.
            window_name: Name identifier string of the cv2 window.

        Raises:
            ValueError: If the cropping parameter is not a valid ROI or tuple of 4 floats.
        """
        # NOTE: Consider splitting the function to return the image and cv2.imshow separately.

        H, W = self.image.shape[:2]
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR) if self.color_format == "RGB" else self.image

        # Display ROI
        if self.roi and self.roi != (0, 0, 1, 1) and self.image_type != IMAGE_TYPE.INPUT_TENSOR:
            left, top, width, height = self.roi
            start_point = (int(left * W), int(top * H))
            end_point = (int((left + width) * W), int((top + height) * H))
            img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)

        # Cropping
        if cropping is not None:
            if not isinstance(cropping, (Tuple, ROI)) or len(cropping) != 4:
                raise ValueError("Cropping must be a tuple of 4 floats or the named tuple ROI.")
            if isinstance(cropping, Tuple):
                cropping = ROI(*cropping)

            if not all(0 <= value <= 1 for value in cropping):
                raise ValueError("All relative cropping values (left, top, width, height) must be between 0 and 1.")
            if cropping.left + cropping.width > 1 or cropping.top + cropping.height > 1:
                raise ValueError("Cropping is out of the frame. Ensure that left + width <= 1 and top + height <= 1.")

            y1 = int(round(cropping.top * H))
            y2 = int(round((cropping.top + cropping.height) * H))
            x1 = int(round(cropping.left * W))
            x2 = int(round((cropping.left + cropping.width) * W))
            img = img[y1:y2, x1:x2]

        if rotate is not None:
            if rotate not in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                raise ValueError(
                    "Invalid rotation code. Use cv2 rotation codes: "
                    "cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180 or cv2.ROTATE_90_COUNTERCLOCKWISE."
                )
            img = cv2.rotate(img, rotate)

        if flip is not None:
            if flip not in [0, 1, -1]:
                raise ValueError("Invalid flip code. Use 0 for vertical, 1 for horizontal, -1 for both.")
            img = cv2.flip(img, flip)

        if show_fps_dps:
            img = cv2.putText(
                img,
                f"FPS: {self.fps:.2f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            img = cv2.putText(
                img,
                f"DPS: {self.dps:.2f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if window_name not in CV2_WINDOWS:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            CV2_WINDOWS.add(window_name)
            cv2.waitKey(100)  # Allow time to create the window

        if resize_image:
            cv2.resizeWindow(window_name, (W, H))
            img = cv2.resize(img, (W, H))

        cv2.imshow(window_name, img)

        # 'ESC' key or window is closed manually
        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            CV2_WINDOWS.clear()
            cv2.destroyAllWindows()
            sys.exit()

    def json(self) -> dict:
        """
        Convert the complete Frame to a JSON-serializable dictionary.

        Returns:
            The JSON-serializable dictionary.
        """
        if self._image is not None:
            ret, buffer = cv2.imencode(
                ".jpg", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR) if self.color_format == "RGB" else self.image
            )
            image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        else:
            image = None

        return {
            "timestamp": self.timestamp,
            "image": image,
            "image_type": self.image_type,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "detections": self.detections.json() if self._detections else None,
            "detection_type": type(self.detections).__name__ if self._detections else None,
            "new_detection": self.new_detection,
            "fps": self.fps,
            "dps": self.dps,
            "color_format": self.color_format,
            "roi": self.roi.json() if self.roi else None,
        }

    @classmethod
    def from_json(cls, data: dict) -> "Frame":
        """
        Create a Frame instance from a JSON-serializable dictionary.

        Args:
            data: JSON-serializable dictionary with frame data.

        Returns:
            The Frame instance created from the JSON data.
        """
        # Decode and decompress the image data if available
        image_data = data.get("image")
        if image_data is not None:
            image = cv2.imdecode(
                np.frombuffer(base64.b64decode(image_data.split(",")[1]), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if data["color_format"] == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = None

        # Get detection results
        detection_type = data.get("detection_type")
        if detection_type is None:
            detections = None
        elif hasattr(RESULT_TYPE, detection_type):
            detection_class = getattr(RESULT_TYPE, detection_type)
            detections = detection_class.from_json(data["detections"])
        else:
            raise TypeError(f"Unsupported detection type: {detection_type}")

        return cls(
            timestamp=data["timestamp"],
            image=image,
            image_type=data["image_type"],
            width=data["width"],
            height=data["height"],
            channels=data["channels"],
            detections=detections,
            new_detection=data["new_detection"],
            fps=data["fps"],
            dps=data["dps"],
            color_format=data["color_format"],
            roi=ROI.from_json(data["roi"]) if data.get("roi") else None,
        )
