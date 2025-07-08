from pathlib import Path
from typing import Optional

from .codecs import FrameCodec, JsonCodec
from ..device import Device
from ..frame import Frame
from modlib.models.model import Model


class Playback(Device):
    """
    Playback device.

    A device capable of playing back recorded frames using a specified frame codec.

    Example usage:
    ```
    device = Playback(recording="./path/to/my_recording.pkl", codec=PickleCodec())

    with device as stream:
        for frame in stream:
            print(frame.detections)
            frame.display()
    ```
    """

    def __init__(
        self,
        recording: str,
        codec: Optional[FrameCodec] = JsonCodec(),
        headless: Optional[bool] = False,
        timeout: Optional[int] = None,
    ):
        self.filepath = Path(recording)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Recording file not found: {self.filepath}")

        # Open file in binary or text mode based on codec
        file_mode = "rb" if codec.binary_mode else "r"
        self.file = self.filepath.open(file_mode, encoding=codec.encoding)
        self.codec = codec

        # Reset file position
        self.file.seek(0)

        super().__init__(
            headless=headless,
            enable_input_tensor=False,
            timeout=timeout,
        )

    def deploy(self, model: Model):
        import warnings

        warnings.warn(
            "Deploying a model to the Playback device will have no affect as the detection results are already captured in the recorded frames.",
            UserWarning,
        )
        # Since playback is based on recorded data, the model is not used.
        pass

    def __enter__(self):
        """
        Start the Playback device stream.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the playback device stream.
        """
        if not self.file.closed:
            self.file.close()

    def __iter__(self):
        """
        Iterate over the frames in the device stream.
        """
        return self

    def __next__(self) -> Frame:
        """
        Get the next frame in the device stream.

        Returns:
            The next frame in the device stream.
        """
        self.check_timeout()

        # Read next frame from the recording
        frame = self.codec.decode(self.file)
        if frame is None:
            raise StopIteration

        return frame
