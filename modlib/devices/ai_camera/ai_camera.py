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

import atexit
import ctypes
import logging
import os
import selectors
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.zoo import InputTensorOnly

from ..device import Device, Rate
from ..frame import IMAGE_TYPE, ROI, Frame
from ..utils import IMX500Converter, check_dir_required
from .allocator import Allocator, DmaAllocator
from .camera_manager import CameraManager
from .imx500 import IMX500
from .libcamera_config import LibcameraConfig
from .rpk_packager import RPKPackager

# Global Libcamera Manager
CM = CameraManager()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s]  %(levelname)s %(message)s")
logger = logging.getLogger(__name__.split(".")[-1])

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

# NOTE: Compatible with libcamera 0.5.0 on Raspberry Pi OS: Bookworm
class _CnnOutputTensorInfoExported(ctypes.LittleEndianStructure):
    _fields_ = [
        ("network_name", ctypes.c_char * NETWORK_NAME_LEN),
        ("num_tensors", ctypes.c_uint32),
        ("info", _OutputTensorInfo * MAX_NUM_TENSORS),
    ]

# NOTE: Compatible with libcamera 0.6.0 on Raspberry Pi OS: Trixie
class _CnnOutputTensorInfoExported2(ctypes.LittleEndianStructure):
    _fields_ = [
        ("network_name", ctypes.c_char * NETWORK_NAME_LEN),
        ("num_tensors", ctypes.c_uint32),
        ("info", _OutputTensorInfo * MAX_NUM_TENSORS),
        ("frameCount", ctypes.c_uint8),
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
        frame_rate: Optional[int] = 30,
        image_size: Tuple[int, int] = (640, 480),
        num: Optional[int] = 0,
    ):
        """
        Initialize the AiCamera device.

        Args:
            headless: Initialising the AiCamera in headless mode means `frame.image` is never processed and unavailable.
            enable_input_tensor: When enabling input tensor, `frame.image` will be replaced by the input tensor image.
            timeout: If set, automatically stop the device loop after the specified seconds.
            frame_rate: The frames per second applied to the libcamera configuration.
            image_size: Resolution of the frame.image. Defaults to (640, 480) which has the original aspect ratio.
            num: The camera number to select which camera to use, when more then one AiCamera connected to libcamera.
        """
        self.config = None
        self.imx500 = None
        self.model = None
        self.roi_it = None  # ROI input tensor
        self.roi_hires = None  # ROI high resolution image
        self.scaler_crop = None
        self.allocator = Allocator()

        # Get the real libcamera internal number (First one listed)
        if len(CM.global_cameras) > 0:
            self.camera_num = CM.global_cameras[num]["Num"]
            self.camera_id = CM.global_cameras[num]["Id"]
            self.camera = CM.cms.cameras[self.camera_num]
        else:
            raise RuntimeError("No camera found by libcamera.")

        # Libcamera request buffer
        # CM notifies when request ready to be processed
        self._requests = []
        self._requestslock = threading.Lock()  # request buffer lock
        self.notifyme_r, self.notifyme_w = os.pipe2(os.O_NONBLOCK)
        self.notifymeread = os.fdopen(self.notifyme_r, "rb")
        self.lock = threading.Lock()  # global request process lock
        self.req_lock = threading.Lock()  # global libcamera request lock

        # Frame buffer
        self._frames = []
        self._frameslock = threading.Lock()  # frame buffer lock
        self._frame_ready = threading.Condition(self._frameslock)
        self.last_detections = None

        # Process thread
        self._proc_started = threading.Event()
        self._proc_abort = threading.Event()
        self._proc_thread = None

        atexit.register(self.stop)
        self._running = False

        self.rps = Rate()
        self.fps = Rate()
        self.dps = Rate()

        self._monitor_abort = threading.Event()
        self._monitor_rate_thread = None

        self.frame_rate = frame_rate
        self.image_size = image_size
        super().__init__(
            headless=headless,
            enable_input_tensor=enable_input_tensor,
            timeout=timeout,
        )

    def _initiate(self, frame_rate, enable_input_tensor, scaler_crop, image_size):
        # 1. Initiate libcamera device & libcamera event listener
        CM.add(self)
        self.camera.acquire()

        # 2. Configure libcamera device
        self.config = LibcameraConfig(self.camera, frame_rate, enable_input_tensor, scaler_crop, image_size)
        libcamera_config = self.config.create()
        if self.camera.configure(libcamera_config):
            raise RuntimeError(f"Libcamera configuration failed: {self.config.camera_config}")

        # 3. Allocate
        self.allocator = DmaAllocator()
        self.allocator.allocate(libcamera_config, self.config.camera_config.get("use_case"))

    def _process_requests_func(self):
        sel = selectors.DefaultSelector()
        sel.register(self.notifyme_r, selectors.EVENT_READ, self._process_requests)
        self._proc_started.set()

        while not self._proc_abort.is_set():
            events = sel.select(0.2)
            for key, _ in events:
                self.notifymeread.read()
                callback = key.data
                callback()

    def _process_requests(self):
        with self._requestslock:
            requests = self._requests
            self._requests = []

        with self.lock:  # for the extracting the image and output tensor
            if not requests:
                return
            req = requests.pop(0)  # possibly more then 1 request in the request buffer to process

            # Process information in the libcamera request
            # And add the processed information as a Frame to the frame_queue
            self._parse_request(req)

            req.release()

            # Release possible remaining requests, those leftovers are thrown away
            for req in requests:
                logger.debug("Processing thread is dropping incoming libcamera requests.")
                req.release()

    def _update_input_tensor_image(self, req):
        input_tensor = req.metadata.get("CnnInputTensor")
        if not input_tensor:
            raise ValueError(
                """
                The provided model was converted with input tensor disabled,
                Provide a model with input tensor enabled.
            """
            )

        w, h = self.model.input_tensor_size
        r1 = np.array(input_tensor, dtype=np.uint8).astype(np.int32).reshape((3,) + (h, w))[(2, 1, 0), :, :]
        norm_val = self.model.info["input_tensor"]["norm_val"]
        norm_shift = self.model.info["input_tensor"]["norm_shift"]
        div_val = self.model.info["input_tensor"]["div_val"]
        div_shift = self.model.info["input_tensor"]["div_shift"]
        for i in [0, 1, 2]:
            r1[i] = ((((r1[i] << norm_shift[i]) - norm_val[i]) << div_shift) // div_val[i]) & 0xFF

        self.input_tensor_image = np.transpose(r1, (1, 2, 0)).astype(np.uint8).copy()

    def _parse_request(self, req):
        # 1. capture-metadata
        output_tensor = req.metadata.get("CnnOutputTensor")
        detections = None
        new_detection = False

        if isinstance(self.model, InputTensorOnly):
            pass
        elif output_tensor:
            np_output = np.fromiter(output_tensor, dtype=np.float32)

            offset = 0
            outputs = []
            for tensor_shape in self._get_output_tensor_shape(req):
                size = np.prod(tensor_shape)
                outputs.append(np_output[offset : offset + size].reshape(tensor_shape, order="F"))
                offset += size

            # Post processing
            detections = self.model.post_process(outputs)

            new_detection = True
            self.last_detections = detections
            self.dps.update()

        elif self.last_detections is None:
            # Missing output tensor in frame & no detection yet
            # Skip adding to frame_queue
            return
        else:
            detections = self.last_detections

        # 2. get VGA or input tensor image
        if self.enable_input_tensor:
            # only update input tensor when available
            if new_detection:
                self._update_input_tensor_image(req)
            image = self.input_tensor_image
        else:
            image = req.image
        h, w, c = image.shape

        with self._frameslock:
            self._frames.append(
                Frame(
                    timestamp=datetime.now().isoformat(),
                    image=image,
                    image_type=IMAGE_TYPE.VGA if not self.enable_input_tensor else IMAGE_TYPE.INPUT_TENSOR,
                    width=w,
                    height=h,
                    channels=c,
                    detections=detections,
                    new_detection=new_detection,
                    fps=self.fps.value,
                    dps=self.dps.value,
                    color_format=COLOR_FORMAT.BGR,  # Both VGA image as input_tensor_image always in BGR format
                    input_tensor=None,
                    roi=self.roi,
                )
            )

            self._frame_ready.notify_all()  # Notify waiting threads

    def _initiate_roi(self) -> None:
        # Full field of view if high res image cropping not specified
        if self.roi_hires is None:
            self.set_image_cropping((0, 0, 1, 1))
        X, Y, W, H = self.roi_hires

        # When input tensor cropping not specified, set according to model requirement
        if self.roi_it is None:
            if self.model.preserve_aspect_ratio:
                model_aspect = self.model.input_tensor_size[0] / self.model.input_tensor_size[1]
                display_aspect = (SENSOR_W * W) / (SENSOR_H * H)
                if model_aspect > display_aspect:
                    w, h = W, display_aspect / model_aspect * H
                else:
                    w, h = model_aspect / display_aspect * W, H
                x, y = X + (W - w) / 2, Y + (H - h) / 2

                self.set_input_tensor_cropping((x, y, w, h))

                logger.warn(
                    f"\033[93mInput tensor cropping settings have been adjusted to preserve aspect ratio of the "
                    f"input tensor (model requirement): ROI({x:g}, {y:g}, {w:g}, {h:g})\033[0m"
                )

            else:
                self.set_input_tensor_cropping((0, 0, 1, 1))

    def _verify_roi(self) -> None:
        # Auto adjust input tensor cropping to fit high res image when needed
        x0, y0, w0, h0 = self.roi_it
        x1, y1, w1, h1 = self.roi_hires

        x2 = max(x0, x1)
        y2 = max(y0, y1)
        w2 = min(x0 + w0, x1 + w1) - x2
        h2 = min(y0 + h0, y1 + h1) - y2

        if not np.allclose((x0, y0, w0, h0), (x2, y2, w2, h2), rtol=1e-6):
            if w2 == 0 or h2 == 0:
                raise ValueError("""
                    Input tensor cropping has no overlapping region with the high resolution image cropping.
                    Please adjust the input tensor cropping settings to fit inside the high resolution image.
                """)

            logger.warn(
                f"\033[93mInput tensor cropping settings have been adjusted to fit inside the high resolution image: "
                f"ROI({x0:g}, {y0:g}, {w0:g}, {h0:g}) -> "
                f"ROI({x2:g}, {y2:g}, {w2:g}, {h2:g})\033[0m"
            )
            self.set_input_tensor_cropping((x2, y2, w2, h2))

        # Setup combined ROI
        self.roi = ROI((x2 - x1) / w1, (y2 - y1) / h1, w2 / w1, h2 / h1)

    def start(self):
        """
        Start the AiCamera device stream.
        """
        if self.model is None:
            self.deploy(InputTensorOnly())

        # Setup & Verify ROI (input tensor & high res image)
        self._initiate_roi()
        self._verify_roi()

        # Initiate libcamera config after the model deployment
        self._initiate(
            frame_rate=self.frame_rate,
            enable_input_tensor=self.enable_input_tensor,
            scaler_crop=self.scaler_crop,
            image_size=self.image_size,
        )

        # Start processing thread
        self._proc_started.clear()
        self._proc_abort.clear()
        self._proc_thread = threading.Thread(target=self._process_requests_func)
        self._proc_thread.setDaemon(True)
        self._proc_thread.start()
        self._proc_started.wait()

        # Start libcamera
        self.camera.start(self.config.libcamera_controls)
        self._running = True
        self.rps.init()
        self.dps.init()

        # Warn when difference between DPS and RPS is large
        self._monitor_abort.clear()
        self._monitor_rate_thread = threading.Thread(target=self._monitor_dps_performance)
        self._monitor_rate_thread.setDaemon(True)
        self._monitor_rate_thread.start()

        for request in self._make_requests():
            self.camera.queue_request(request)

        logger.info("Camera Started !")

    def _make_requests(self):
        num_requests = min([len(self.allocator.buffers(stream)) for stream in self.config.streams])
        requests = []
        for i in range(num_requests):
            request = self.camera.create_request(self.camera_num)
            if request is None:
                raise RuntimeError("Could not create request")

            for stream in self.config.streams:
                # This now throws an error if it fails.
                request.add_buffer(stream, self.allocator.buffers(stream)[i])
            requests.append(request)
        return requests

    def _monitor_dps_performance(self):
        # Wait for initial 10 seconds to let the system stabilize
        for _ in range(20):
            if self._monitor_abort.is_set():
                return
            time.sleep(0.5)

        # Check rate difference
        THRESHOLD = 5
        if abs(self.rps.value - self.dps.value) > THRESHOLD:
            logger.warning(
                f"\033[93mPerformance warning: Large difference between libcamera request rate ({self.rps.value:.1f} RPS) "
                f"and detection rate ({self.dps.value:.1f} DPS). "
                f"Consider lowering the frame rate for better DPS performance. E.g. `AiCamera(frame_rate=<inbetween RPS and DPS value>)`\033[0m"
            )

    def stop(self):
        """
        Stop the AiCamera device stream.
        """
        atexit.unregister(self.stop)

        if self.imx500 is not None:
            self.imx500.stop_network_fw_progress_bar()

        # Stop processing thread
        if self._proc_thread and self._proc_thread.is_alive():
            self._proc_abort.set()
            self._proc_thread.join()

        if self._monitor_rate_thread and self._monitor_rate_thread.is_alive():
            self._monitor_abort.set()
            self._monitor_rate_thread.join()

        if self._running:
            self._running = False
            self.camera.stop()

            # Clear unseen requests
            CM.handle_request(self.camera_num)
            self.camera.release()
            with self._requestslock:
                unseen_requests = self._requests
                self._requests.clear()
            for r in unseen_requests:
                r.release()

            CM.cleanup(self.camera_num)

        del self.imx500
        self.camera_num = None
        self.camera_id = None
        self.camera = None
        self.config = None
        self.imx500 = None
        self.model = None

        self.notifymeread.close()
        os.close(self.notifyme_w)

        # Clear any frames in the framebuffer before deleting the annotator
        with self._frameslock:
            self._frames.clear()

        del self.allocator
        self.allocator = Allocator()
        logger.info("Camera closed successfully.")

    def set_input_tensor_cropping(self, roi: Union[ROI, Tuple[float, float, float, float]]):
        """
        Set the input tensor cropping.

        Args:
            roi: The relative ROI (region of interest) in the form a (left, top, width, height) [%] crop for
                the input inference.
        """
        if not isinstance(roi, (Tuple, ROI)) or len(roi) != 4:
            raise ValueError("roi must be a tuple of 4 floats or the named tuple ROI.")
        if isinstance(roi, Tuple):
            roi = ROI(*roi)
        self.roi_it = roi

        if not self.model:
            raise ValueError("No model deployed. Make sure to deploy a model before setting the input tensor cropping.")
        if not all(0 <= value <= 1 for value in roi):
            raise ValueError("All relative ROI values (left, top, width, height) must be between 0 and 1.")

        (left, top, width, height) = roi
        if left + width > 1 or top + height > 1:
            raise ValueError("ROI is out of the frame. Ensure that left + width <= 1 and top + height <= 1.")

        # Convert to absolute ROI based on full sensor resolution
        roi_abs = (int(left * SENSOR_W), int(top * SENSOR_H), int(width * SENSOR_W), int(height * SENSOR_H))
        self.imx500.set_inference_roi_abs(roi_abs)
        if self._running:
            self._verify_roi()

    def set_image_cropping(self, roi: Union[ROI, Tuple[float, float, float, float]]):
        """
        Set the cropping of the high resolution image. Can only be adjusted during initialisation.

        Args:
            roi: The relative ROI (region of interest) in the form a (left, top, width, height) [%] crop.
        """
        if self._running:
            raise RuntimeError("Cannot adapt the camera high res image cropping while running.")

        if not isinstance(roi, (Tuple, ROI)) or len(roi) != 4:
            raise ValueError("roi must be a tuple of 4 floats or the named tuple ROI.")
        if isinstance(roi, Tuple):
            roi = ROI(*roi)
        self.roi_hires = roi

        (left, top, width, height) = roi
        if not all(0 <= value <= 1 for value in roi):
            raise ValueError("All ROI values (left, top, width, height) must be between 0 and 1.")
        if left + width > 1 or top + height > 1:
            raise ValueError("ROI is out of the frame. Ensure that left + width <= 1 and top + height <= 1.")

        self.scaler_crop = (int(left * SENSOR_W), int(top * SENSOR_H), int(width * SENSOR_W), int(height * SENSOR_H))
        # if self._running: # NOTE: unnecessary as you can't adjust while running and verify is called at start()
        #     self._verify_roi()

    # Model deployment
    def prepare_model_for_deployment(self, model: Model, overwrite: Optional[bool] = None) -> str | None:
        """
        Prepares a model for deployment by converting and/or packaging it based on the model type.
        Behaviour of the deployment depends on model type:
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
            overwrite: If None, prompts the user for input. If True, overwrites the output directory if it exists.
                If False, uses already converted/packaged model from the output directory.
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
                logger.error(f"Caught an assertion error: {e}")
                return None

        # packaged model - done
        if model.model_type == MODEL_TYPE.RPK_PACKAGED:
            network_file = model.model_file
            logger.info(f"Packaged model: {network_file}")

        # converted model - package
        elif model.model_type == MODEL_TYPE.CONVERTED:
            network_file = package()
            logger.info(f"Converted model: {network_file}")

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
                logger.info(f"Missing file: {network_file}")
                network_file = None

        logger.info(f"network_file: {network_file}")
        return network_file

    def deploy(self, model: Model, overwrite: Optional[bool] = None) -> None:
        """
        This method manages the process to run a model on the device. This requires the
        following steps:

        - Prepare model for deployment
        - Configure model deployment

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

        # configure model deployment
        self.model = model
        self.model._get_network_info(Path(network_file))
        self.imx500 = IMX500(os.path.abspath(network_file), camera_id=self.camera_id)
        self.imx500.show_network_fw_progress_bar()

    def get_device_id(self) -> str | None:
        """
        Retrieve the unique IMX500 device ID.

        Returns:
            The device ID as an ASCII string if successful, otherwise None.
        """
        return self.imx500.get_device_id()

    @staticmethod
    def _get_output_tensor_shape(req):
        # TODO: can be removed when output tensor shape available in model
        output_tensor_info = req.metadata.get("CnnOutputTensorInfo")
        if not output_tensor_info:
            return []

        if type(output_tensor_info) not in [bytes, bytearray]:
            output_tensor_info = bytes(output_tensor_info)
        
        # NOTE: Compatible with both libcamera 0.5.0 and 0.6.0
        if len(output_tensor_info) == ctypes.sizeof(_CnnOutputTensorInfoExported):
            parsed = _CnnOutputTensorInfoExported.from_buffer_copy(output_tensor_info)
        elif len(output_tensor_info) == ctypes.sizeof(_CnnOutputTensorInfoExported2):
            parsed = _CnnOutputTensorInfoExported2.from_buffer_copy(output_tensor_info)
        else:
            raise ValueError(f"tensor info length {len(output_tensor_info)} does not match expected size")
        
        return [list(t.size)[: t.num_dimensions] for t in parsed.info[: parsed.num_tensors]]

    def __enter__(self):
        """
        Start the AiCamera device stream.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the AiCamera device stream.
        """
        self.stop()

    def __iter__(self):
        """
        Iterate over the frames in the device stream.
        """
        self.fps.init()
        return self

    def get_frame(self) -> Frame:
        """
        Gets the next processed frame in the device stream.

        Returns:
            The next frame in the device stream.
        """
        with self._frameslock:
            while not self._frames:
                self._frame_ready.wait()  # Wait for available frame

            # We only allow the main thread to pop and clear the frame buffer
            # Calling get_frame() from a thread other than the main thread is allowed and
            # returns the last frame in the buffer for processing
            if threading.current_thread() is threading.main_thread():
                frame = self._frames.pop(-1)
                if self._frames:
                    logger.debug(f"Main thread is dropping {len(self._frames)} frames.")
                    self._frames.clear()
                self.fps.update()
            else:
                frame = self._frames[-1]

        return frame

    def __next__(self) -> Frame:
        """
        Get the next frame in the device stream.

        Returns:
            The next frame in the device stream.
        """
        self.check_timeout()
        return self.get_frame()
