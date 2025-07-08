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

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np


class Source(ABC):
    """
    Abstract base class for input stream sources.
    This class defines a common interface for various types of device stream sources.
    """

    width: int  #: Width of the frames provided by the source.
    height: int  #: Height of the frames provided by the source.
    channels: int  #: Number of channels in the frames provided by the source.
    color_format: str  #: The color format of the frames provided by the source.

    @abstractmethod
    def get_frame(self) -> np.ndarray | None:
        """
        Abstract method to retrieve the next frame image from the source.

        Returns:
            The next frame as an image array or None if no more frames are available.
        """
        pass

    @abstractmethod
    def timestamp(self) -> datetime:
        """
        Abstract method to retrieve the timestamp attached to the current indexed frame.

        Returns:
            The datetime of the frame.
        """
        pass


class Images(Source):
    """
    Source for images.

    Example:
    ```
    from modlib.devices import Images, KerasInterpreter

    device = KerasInterpreter(source=Images("./path/to/image_dir"))

    with device as stream:
        for frame in stream:
            frame.display()
    ```
    """

    def __init__(self, images_dir: Path):
        """
        Initialize an Image source.

        Args:
            images_dir: Path to the directory containing jpg/jpeg/png images.

        Raises:
            FileNotFoundError: When the provided directory does not exist.
            FileNotFoundError: When no images were found in the directory.
        """
        images_dir = Path(images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"\nThe directory {images_dir} does not exist.\n")

        self.image_files = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        )

        if not self.image_files:
            raise FileNotFoundError(f"\nNo image files found in the directory {images_dir}.\n")

        self.image_number = 0
        self.width = None
        self.height = None
        self.channels = None
        self.color_format = "BGR"

    def get_frame(self) -> np.ndarray | None:
        """
        Retrieve the next image from the provided image directory.

        Returns:
            The next image as an image array or None if no more images are available.
        """
        if self.image_number >= len(self.image_files):
            return None

        image = cv2.imread(str(self.image_files[self.image_number]))

        self.height, self.width, self.channels = image.shape
        self.image_number += 1

        return image

    @property
    def timestamp(self):
        """
        Returns:
            Current datetime.
        """
        return datetime.now()


class Video(Source):
    """
    Source for video files.

    Example:
    ```
    from modlib.devices import KerasInterpreter, Video

    device = KerasInterpreter(source=Video("./path/to/video.mp4"))

    with device as stream:
        for frame in stream:
            frame.display()
    ```
    """

    def __init__(self, video_path: Path):
        """
        Initialize a Video source.

        Args:
            video_path: Path to the video file.

        Raises:
            FileNotFoundError: When the provided video_path does not exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"\nThe file {video_path} does not exist.\n")

        self.cap = cv2.VideoCapture(os.path.abspath(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_number = 0
        self.channels = 3
        self.color_format = "BGR"

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.start_time = datetime.now()

    def get_frame(self) -> np.ndarray | None:
        """
        Retrieve the next image from the provided video stream.

        Returns:
            The next image as an image array or None if the full video has been completed.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        _, image = self.cap.read()
        self.frame_number += 1
        return image

    @property
    def timestamp(self):
        """
        Get the timestamp attached to the current indexed frame.
        Calculated as:
            initialization_start_time + (current_frame_number / video_fps)

        Returns:
            The datetime of the frame.
        """
        return self.start_time + timedelta(seconds=self.frame_number / self.fps)
