import base64
import copy
import cv2
import json
import pickle
import numpy as np

from abc import ABC, abstractmethod
from typing import IO, BinaryIO, Union

from ..frame import Frame


class FrameCodec(ABC):
    @staticmethod
    @abstractmethod
    def encode(frame: Frame, file: Union[IO, BinaryIO]):
        """
        Encode a frame and write it to a file

        Args:
            frame: The frame to encode
            file: The open file object to write to
        """
        pass

    @staticmethod
    @abstractmethod
    def decode(file: Union[IO, BinaryIO]) -> Frame | None:
        """
        Decode a frame from a file

        Args:
            file: The open file object to read from

        Returns:
            The decoded frame, or None at the end of the recording.
        """
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Returns the file extension (e.g., 'json', 'pkl', 'bin')."""
        pass

    @property
    @abstractmethod
    def binary_mode(self) -> bool:
        """Returns True if the file open mode is binary, otherwise False."""
        pass

    @property
    def encoding(self) -> str | None:
        """Returns the encoding if applicable, otherwise None."""
        return None  # Default to None for binary codecs


class JsonCodec(FrameCodec):
    """
    A codec for encoding and decoding frames using (human-readable) JSON format.
    """

    @staticmethod
    def encode(frame: Frame, file: Union[IO, BinaryIO]):
        json.dump(frame.json(), file)
        file.write("\n")

    @staticmethod
    def decode(file: Union[IO, BinaryIO]) -> Frame | None:
        line = file.readline()
        if not line:
            return None
        return Frame.from_json(json.loads(line))

    @property
    def file_extension(self) -> str:
        return "json"

    @property
    def binary_mode(self) -> bool:
        return False

    @property
    def encoding(self) -> str | None:
        return "utf-8"


class PickleCodec(FrameCodec):
    """
    A codec for encoding and decoding frames using Python's pickle format.
    """

    @staticmethod
    def encode(frame: Frame, file: Union[IO, BinaryIO]):
        frame_copy = copy.copy(frame)

        # Image encoding
        if frame._image is not None:
            ret, buffer = cv2.imencode(
                ".jpg", cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR) if frame.color_format == "RGB" else frame.image
            )
            frame_copy.image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        # Serialize and write as a single line
        pickle.dump(frame_copy, file)
        file.flush()

    @staticmethod
    def decode(file: Union[IO, BinaryIO]) -> Frame | None:
        # Check if there is more data to read
        current_position = file.tell()
        file.seek(0, 2)
        end_position = file.tell()
        file.seek(current_position)
        if current_position == end_position:
            return None

        frame = pickle.load(file)

        # Decode and decompress the image data if available
        if frame.image is not None:
            frame.image = cv2.imdecode(
                np.frombuffer(base64.b64decode(frame.image.split(",")[1]), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if frame.color_format == "RGB":
                frame.image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

        return frame

    @property
    def file_extension(self) -> str:
        return "pkl"

    @property
    def binary_mode(self) -> bool:
        return True

    @property
    def encoding(self) -> str | None:
        return None
