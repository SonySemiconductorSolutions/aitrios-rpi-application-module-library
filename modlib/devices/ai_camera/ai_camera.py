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

import ctypes
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.results import Anomaly, Classifications, Detections, Poses, Segments
from modlib.models.zoo import InputTensorOnly

from ..device import Device
from ..frame import IMAGE_TYPE, ROI, Frame
from ..utils import IMX500Converter, check_dir_required
from .rpk_packager import RPKPackager

SENSOR_W = 4056
SENSOR_H = 3040

NETWORK_NAME_LEN = 64
MAX_NUM_TENSORS = 16
MAX_NUM_DIMENSIONS = 16


class _OutputTensorInfo(ctypes.LittleEndianStructure):
    _fields_ = [
        ("tensor_data_num", ctypes.c_uint32),
        ("num_dimensions", ctypes.c_uint32),
        ("size", ctypes.c_uint16 * MAX_NUM_DIMENSIONS),
    ]


class _CnnOutputTensorInfoExported(ctypes.LittleEndianStructure):
    _fields_ = [
        ("network_name", ctypes.c_char * NETWORK_NAME_LEN),
        ("num_tensors", ctypes.c_uint32),
        ("info", _OutputTensorInfo * MAX_NUM_TENSORS),
    ]


class AiCamera(Device):
    """
    The Raspberry Pi AI Camera.

    This camera device module allows to run model inference on the IMX500 vision sensor.
    Output tensors are post-processed by the model post-processor function and attached to the frame.

    Example:
    ```
    from modlib.devices import AiCamera
    from modlib.models.zoo import SSDMobileNetV2FPNLite320x320

    device = AiCamera()
    model = SSDMobileNetV2FPNLite320x320()
    device.deploy(model)

    with device as stream:
        for frame in stream:
            print(frame.detections)
    ```
    """

    def __init__(
        self,
        headless: Optional[bool] = False,
        enable_input_tensor: Optional[bool] = False,
        timeout: Optional[int] = None,
    ):
        """
        Initialize the AiCamera device.

        Args:
            headless: Initialising the AiCamera in headless mode means `frame.image` is never processed and unavailable.
            enable_input_tensor: When enabling input tensor, `frame.image` will be replaced by the input tensor image.
            timeout: If set, automatically stop the device loop after the specified seconds.
        """
        self.model = None
        self.picam2 = None
        self.imx500_model = None
        self.roi_relative = None
        self.image = None

        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()

        self.last_detections = None
        self.last_input_tensor = None
        self.detection_times = deque(maxlen=30)
        self.last_detection_time = time.perf_counter()
        self.dps = 0

        super().__init__(
            headless=headless,
            enable_input_tensor=enable_input_tensor,
            timeout=timeout,
        )

    # <------------ Entrypoints ------------>
    def set_input_tensor_cropping(self, roi_relative: Union[ROI, Tuple[float, float, float, float]]):
        """
        Set the input tensor cropping.

        Args:
            roi_relative: The relative ROI (region of interest) in the form a (left, top, width, height) [%] crop for
                the input inference.
        """
        if not isinstance(roi_relative, (Tuple, ROI)) or len(roi_relative) != 4:
            raise ValueError("roi_relative must be a tuple of 4 floats or the named tuple ROI.")
        if isinstance(roi_relative, Tuple):
            roi_relative = ROI(*roi_relative)
        self.roi_relative = roi_relative

        if not self.model:
            raise ValueError("No model deployed. Make sure to deploy a model before setting the input tensor cropping.")
        if not all(0 <= value <= 1 for value in roi_relative):
            raise ValueError("All relative ROI values (left, top, width, height) must be between 0 and 1.")

        (left, top, width, height) = roi_relative
        if left + width > 1 or top + height > 1:
            raise ValueError("ROI is out of the frame. Ensure that left + width <= 1 and top + height <= 1.")

        # Convert to absolute ROI based on full sensor resolution
        roi_abs = (int(left * SENSOR_W), int(top * SENSOR_H), int(width * SENSOR_W), int(height * SENSOR_H))
        self.imx500_model.set_inference_roi_abs(roi_abs)

    def prepare_model_for_deployment(self, model: Model, overwrite: Optional[bool] = None) -> str | None:
        """
        Prepares a model for deployment by converting and/or packaging it based on the model type.
        Behaviour of the deployement depends on model type:
        - RPK_PACKAGED: The model is already packaged, so the path is returned as is.
        - CONVERTED: The model is a converted file (e.g., packerOut.zip), which must be packaged before deployment.
        - KERAS or ONNX: Framework model files, which must be converted and then packaged.
        - If the model type is unsupported or the file doesn't exist after processing, None is returned.

        Args:
            model: The model to be prepared. Can be of various types such as ONNX, KERAS, CONVERTED, or RPK_PACKAGED.
            overwrite: If None, prompts the user for input. If True, overwrites the output directory if it exists.
                If False, uses already converted/packaged model from the output directory.

        Returns:
            The path to the packaged model file ready for deployment. Returns None if the process fails.
        """

        def package() -> str | None:
            """
            Packages the model file using the provided packager and checks the directory for the required output.

            This method runs the packaging process for the model by creating a "pack" subdirectory,
            and then uses the `packager` to generate a deployment-ready `.rpk` file. After running
            the packaging process, it ensures that the expected `.rpk` file exists in the output directory.

            Returns:
                The path to the packaged `.rpk` file if successful, otherwise None if an error occurred.
            """
            packager = RPKPackager()
            d = os.path.dirname(model.model_file)
            pack_dir = os.path.join(d, "pack")

            packager.run(
                input_path=(
                    model.model_file
                    if model.model_type == MODEL_TYPE.CONVERTED
                    else os.path.join(pack_dir, "packerOut.zip")
                ),
                output_dir=pack_dir,
                color_format=model.color_format,
                overwrite=overwrite,
            )
            try:
                check_dir_required(pack_dir, ["network.rpk"])
                return os.path.join(pack_dir, "network.rpk")
            except AssertionError as e:
                print(f"Caught an assertion error: {e}")
                return None

        # packaged model - done
        if model.model_type == MODEL_TYPE.RPK_PACKAGED:
            network_file = model.model_file
            print(f"Packaged model: {network_file}")

        # converted model - package
        elif model.model_type == MODEL_TYPE.CONVERTED:
            network_file = package()
            print(f"Converted model: {network_file}")

        # framework model - convert and package
        elif model.model_type == MODEL_TYPE.KERAS or model.model_type == MODEL_TYPE.ONNX:
            converter = IMX500Converter()
            d = os.path.dirname(model.model_file)
            pack_dir = os.path.join(d, "pack")
            converter.run(
                model_file=model.model_file,
                model_type=model.model_type,
                output_dir=pack_dir,
                overwrite=overwrite,
            )
            network_file = package()

        # oops
        else:
            network_file = None

        # We always make sure that network_file exists at the end
        # It can fail both in converter and in packager
        if network_file is not None:
            if not os.path.exists(network_file):
                print(f"Missing file: {network_file}")
                network_file = None

        print(f"network_file: {network_file}")
        return network_file

    def _configure_for_deployment(self, model: Model, network_file: Path) -> None:
        # TODO: how to unit test? Several of these function raise exceptions
        # TODO: preserve_aspect_ratio - this is not the correct place for something that is not device specific.

        self.model = model
        self.model._get_network_info(network_file)
        self.imx500_model = self._get_imx500_model(network_file)
        self.imx500_model.show_network_fw_progress_bar()

        if model.preserve_aspect_ratio:
            model_aspect = self.model.input_tensor_size[0] / self.model.input_tensor_size[1]
            sensor_aspect = SENSOR_W / SENSOR_H
            if model_aspect > sensor_aspect:
                w, h = 1, sensor_aspect / model_aspect
            else:
                w, h = model_aspect / sensor_aspect, 1
            self.set_input_tensor_cropping(((1 - w) / 2, (1 - h) / 2, w, h))
        else:
            self.set_input_tensor_cropping((0, 0, 1, 1))

    def deploy(self, model: Model, overwrite: Optional[bool] = None) -> None:
        """
        This method manages the process to run a model on the device. This requires the
        following steps:

        - Prepare model for deployment
        - Configure deployment
        - Start the camera with the model

        Args:
            model: The model to be deployed on the device.
            overwrite: If None, prompts the user for input. If True, overwrites the output directory if it exists.
                If False, uses already converted/packaged model from the output directory.

        Raises:
            FileNotFoundError: If the packaged network file cannot be found.
        """
        # Prepare model
        network_file = self.prepare_model_for_deployment(model, overwrite)
        if network_file is None:
            raise FileNotFoundError("Packaged network file error")

        # configure deployment
        self._configure_for_deployment(model, Path(network_file))

        # start camera
        # Initiate Picamera2 (reads the symlink)
        self.picam2 = self._initiate_picamera2()
        self._picam2_start()

    def _picam2_start(self):

        config = self.picam2.create_preview_configuration(
            main={"format": "RGB888"},
            controls={"FrameRate": 30, "CnnEnableInputTensor": self.enable_input_tensor},
            buffer_count=28,
        )

        def pre_callback(request):
            from picamera2 import MappedArray

            # Get VGA image from pre-callback (when enable_input_tensor=False)
            with MappedArray(request, "main") as m:
                self.image = m.array.copy()
                self.height, self.width, self.num_channels = np.shape(self.image)

        self.picam2.start(config, show_preview=False)
        if not self.enable_input_tensor:
            self.picam2.pre_callback = pre_callback

    def _get_output_shapes(self, metadata: dict) -> List[List[int]]:
        """
        Get the model output shapes if no output return empty list
        """
        # TODO: can be removed when output tensor shape available in model

        output_tensor_info = metadata.get("CnnOutputTensorInfo")
        if not output_tensor_info:
            return []

        if type(output_tensor_info) not in [bytes, bytearray]:
            output_tensor_info = bytes(output_tensor_info)
        if len(output_tensor_info) != ctypes.sizeof(_CnnOutputTensorInfoExported):
            raise ValueError(f"tensor info length {len(output_tensor_info)} does not match expected size")

        parsed = _CnnOutputTensorInfoExported.from_buffer_copy(output_tensor_info)
        # Getting the output tensor shapes only
        return [list(t.size)[: t.num_dimensions] for t in parsed.info[: parsed.num_tensors]]

    def _get_input_tensor_image(self, metadata: dict) -> np.ndarray:
        """
        Get and convert input tensor to BGR image format.
        """
        input_tensor = metadata.get("CnnInputTensor")
        if not input_tensor:
            raise ValueError(
                """
                The provided model was converted with input tensor disabled,
                Provide a model with input tensor enabled.
            """
            )

        # NOTE: this info is also available in self.model_get_network_info()
        width = self.imx500_model.config["input_tensor"]["width"]
        height = self.imx500_model.config["input_tensor"]["height"]
        r1 = np.array(input_tensor, dtype=np.uint8).astype(np.int32).reshape((3,) + (height, width))[(2, 1, 0), :, :]
        norm_val = self.imx500_model.config["input_tensor"]["norm_val"]
        norm_shift = self.imx500_model.config["input_tensor"]["norm_shift"]
        div_val = self.imx500_model.config["input_tensor"]["div_val"]
        div_shift = self.imx500_model.config["input_tensor"]["div_shift"]
        for i in [0, 1, 2]:
            r1[i] = ((((r1[i] << norm_shift[i]) - norm_val[i]) << div_shift) // div_val[i]) & 0xFF

        return np.transpose(r1, (1, 2, 0)).astype(np.uint8).copy()

    def _picam2_thread_function(self, queue, model):

        if model is None:
            self.deploy(InputTensorOnly())

        while not self.stop_event.is_set():
            try:
                metadata = self.picam2.capture_metadata()
                output_tensor = metadata.get("CnnOutputTensor")

                detections = None
                new_detection = False
                input_tensor = None

                # Process output tensor
                if model is None:
                    if self.enable_input_tensor:
                        self.image = self._get_input_tensor_image(metadata)
                        self.height, self.width, self.num_channels = np.shape(self.image)

                elif output_tensor:
                    # reshape buffer to tensor shapes
                    # TODO: reavaluate when output tensor shape available in model
                    np_output = np.fromiter(output_tensor, dtype=np.float32)
                    output_shapes = self._get_output_shapes(metadata)

                    offset = 0
                    outputs = []
                    for tensor_shape in output_shapes:
                        size = np.prod(tensor_shape)
                        outputs.append(np_output[offset : offset + size].reshape(tensor_shape, order="F"))
                        offset += size

                    # Post processing
                    detections = model.post_process(outputs)

                    new_detection = True
                    self.last_detections = detections
                    self._update_dps()

                    # Get input tensor if enabled
                    if self.enable_input_tensor:
                        # 1. Setting frame.image to input tensor image
                        self.image = self._get_input_tensor_image(metadata)
                        self.height, self.width, self.num_channels = np.shape(self.image)

                        # 2. getting the real ISP output (for framework input)
                        # TODO
                        input_tensor = np.empty((0,))
                        self.last_input_tensor = input_tensor

                elif self.last_detections is None:
                    # Missing output tensor in frame (no detection yet)
                    continue
                else:
                    # Missing output tensor in frame
                    detections = self.last_detections
                    input_tensor = self.last_input_tensor

                # Append frame to frame queue
                queue.put(
                    Frame(
                        timestamp=datetime.now().isoformat(),
                        image=self.image,
                        image_type=IMAGE_TYPE.VGA if not self.enable_input_tensor else IMAGE_TYPE.INPUT_TENSOR,
                        width=self.width,
                        height=self.height,
                        channels=self.num_channels,
                        detections=detections,
                        new_detection=new_detection,
                        fps=self.fps,
                        dps=self.dps,
                        color_format=COLOR_FORMAT.BGR,  # Both VGA image as input_tensor_image always in BGR format
                        input_tensor=input_tensor,
                        roi=self.roi_relative,
                    )
                )

            except KeyError:
                pass

        self.picam2.close()

    def _update_dps(self):
        current_time = time.perf_counter()
        self.detection_times.append(current_time - self.last_detection_time)
        self.last_detection_time = current_time
        if len(self.detection_times) > 1:
            self.dps = len(self.detection_times) / sum(self.detection_times)

    # <------------ Stream ------------>
    def __enter__(self):
        """
        Start the AiCamera device stream.
        """
        self.stop_event.clear()

        self.picam2_thread = threading.Thread(target=self._picam2_thread_function, args=(self.frame_queue, self.model))
        self.picam2_thread.start()

        self.start_time = time.perf_counter()
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the AiCamera device stream.
        """
        self.stop_event.set()
        self.picam2_thread.join()

    def __iter__(self):
        """
        Iterate over the frames in the device stream.
        """
        self.last_time = time.perf_counter()
        return self

    def __next__(self) -> Frame:
        """
        Get the next frame in the device stream.

        Returns:
            The next frame in the device stream.
        """
        self.check_timeout()
        self.update_fps()

        try:
            return self.frame_queue.get(timeout=120)
        except queue.Empty:
            raise StopIteration

    @staticmethod
    def _initiate_picamera2():
        try:
            from picamera2 import Picamera2

            return Picamera2()
        except ImportError:
            raise ImportError(
                """
                picamera2 is not installed. Please install picamera2 to use the AiCamera device.\n\n
                For a raspberry pi with picamera2 installed. Enable in virtual env using:
                `python -m venv .venv --system-site-packages`\n
                """
            )

    @staticmethod
    def _get_imx500_model(model_path):
        try:
            from picamera2.devices.imx500 import IMX500

            return IMX500(os.path.abspath(model_path))
        except ImportError:
            raise ImportError(
                """
                picamera2 is not installed. Please install picamera2 to use the AiCamera device.\n\n
                For a raspberry pi with picamera2 installed. Enable in virtual env using:
                `python -m venv .venv --system-site-packages`\n
                """
            )
