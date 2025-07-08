import atexit
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import IO, BinaryIO, Union

from .codecs import FrameCodec, JsonCodec
from ..frame import Frame


class Recorder:
    """
    Recorder class to record a series of frames to a file using a specified frame codec.

    Example usage:
    ```
    rec = Recorder(directory='./temp/recordings', codec=PickleCodec())

    with device as stream:
        for frame in stream:
            rec.add(frame)  # Add the frame to the recording
    ```
    """

    codec: FrameCodec  #: The codec used for encoding/decoding frames
    path: Path  #: Recording file path
    file: Union[IO, BinaryIO]  #: The open recording file object (automatically closed on exit)

    MIN_FREE_SPACE: int = 100 * 1024 * 1024  # 100MB buffer to prevent filling disk

    def __init__(self, directory: str, codec: FrameCodec = JsonCodec()):
        if not isinstance(codec, FrameCodec):
            raise ValueError("Codec must be an instance of FrameCodec.")
        self.codec = codec

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"recording_{timestamp}.{codec.file_extension}"
        self.path = directory / filename

        # Check if file already exists
        if self.path.exists():
            raise FileExistsError(f"Recording file already exists: {self.path}")

        # Open file using codec-defined mode and encoding
        file_model = "ab" if codec.binary_mode else "a"
        self.file = self.path.open(file_model, encoding=codec.encoding)

        atexit.register(self.close)

    def close(self):
        """Close the recording file."""
        if not self.file.closed:
            self.file.close()

    def add(self, frame: Frame):
        """Save a frame to the recording file."""

        free_space = shutil.disk_usage(self.path.parent).free
        if free_space < self.MIN_FREE_SPACE:
            warnings.warn(
                f"Warning: Low disk space ({free_space / (1024 * 1024)} MB available). Stopping recording.",
                UserWarning,
            )
            self.close()
            sys.exit()

        self.codec.encode(frame, self.file)
