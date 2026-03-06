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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from modlib.models import COLOR_FORMAT


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
        ...

    @abstractmethod
    def timestamp(self) -> datetime:
        """
        Abstract method to retrieve the timestamp attached to the current indexed frame.

        Returns:
            The datetime of the frame.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        Abstract method to retrieve the number of frames/images in the source.

        Returns:
            The number of frames/images in the source.
        """
        ...

    def __iter__(self):
        """
        Make the Source iterable.

        Returns:
            The source itself, which can be iterated over.
        """
        return self

    def __next__(self) -> np.ndarray:
        """
        Get the next frame from the source.

        Returns:
            The next frame as an image array.

        Raises:
            StopIteration: When no more frames are available.
        """
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame


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

        self.image_number = 0
        self.width = None
        self.height = None
        self.channels = None
        self.color_format = COLOR_FORMAT.BGR

    def get_frame(self) -> np.ndarray | None:
        """
        Retrieve the next image from the provided image directory.

        Returns:
            The next image as an image array or None if no more images are available.
        """
        if self.image_number >= len(self.image_files):
            return None

        # NOTE: always BGR
        image = cv2.imread(str(self.image_files[self.image_number]))

        self.height, self.width, self.channels = image.shape
        self.image_number += 1

        return image

    def __len__(self) -> int:
        return len(self.image_files)

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
        self.color_format = COLOR_FORMAT.BGR

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

    def __len__(self) -> int:
        return self.total_frames

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


@dataclass
class DatasetSample:
    """
    Container for dataset sample metadata and image content.
    """

    image_path: Path  #: Path to the image.
    image_name: str  #: Name of the image.
    image_id: Optional[int | str]  #: Image ID. Supports both int (COCO) and str (VOC).
    image: np.ndarray  #: The sample image.
    image_color_format: COLOR_FORMAT  #: Color format of the sample image.


class Dataset(Source):
    """
    Source that walks a directory tree and yields dataset samples.
    Useful for evaluation pipelines that need specific image ids from file names or folder structures.
    """

    def __init__(
        self,
        images_dir: Path,
        dataset_id_function: Optional[Callable] = None,
    ):
        """
        Build a dataset source.

        Args:
            images_dir: Root directory scanned recursively for images.
            dataset_id_function: Optional callable mapping a Path to an image id; defaults to the file stem.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"\nThe directory {self.images_dir} does not exist.\n")
        self.dataset_id_function = dataset_id_function

        # Get all image files in the directory and its subdirectories (recursive)
        # Classification uses folder-based structure (class_id/image.jpg)
        # Detection/segmentation may use flat structure
        self.image_files = []
        for ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            self.image_files += list(self.images_dir.rglob(f"*{ext}"))
        self.image_files = sorted(self.image_files)

        self.image_number = 0
        self.width = None
        self.height = None
        self.channels = None
        self.color_format = COLOR_FORMAT.BGR  # Color format returned by get_frame()

    def get_frame(self) -> np.ndarray | None:
        """
        Load the next image from the dataset.

        Returns:
            Image array in BGR format, or None when all images are consumed.
        """
        if self.image_number >= len(self.image_files):
            return None

        image = cv2.imread(str(self.image_files[self.image_number]))  # NOTE: always BGR
        self.height, self.width, self.channels = image.shape
        self.image_number += 1

        return image

    def get_image_id(self, img_path: Path) -> int | str:
        """
        Resolve the image id for a given path.

        Args:
            img_path: Path to the image file.

        Returns:
            Id from `dataset_id_function` or the file stem (cast to int when numeric).
        """
        if self.dataset_id_function is None:
            return int(img_path.stem) if img_path.stem.isdigit() else img_path.stem
        return self.dataset_id_function(img_path)

    def __iter__(self) -> Iterator[DatasetSample]:
        """
        Iterate over dataset samples.

        Yields:
            DatasetSample with path, name, id, and image in RGB format.
        """
        for img_path in self.image_files:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue  # skip corrupted/not found images
            yield DatasetSample(
                image_path=img_path,
                image_name=img_path.name,
                image_id=self.get_image_id(img_path),
                image=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                image_color_format=COLOR_FORMAT.RGB,
            )

    def __len__(self) -> int:
        """Number of images discovered in the dataset."""
        return len(self.image_files)

    @property
    def timestamp(self):
        """
        Returns:
            Current datetime.
        """
        return datetime.now()
