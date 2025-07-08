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
import atexit
import ctypes
import logging
import multiprocessing
import threading
import signal
import numpy as np
import cv2
import time
from datetime import datetime
from typing import Optional, Tuple, Union


from ..device import Device, Rate
from ..frame import IMAGE_TYPE, ROI, Frame
from .request_interface import TritonConfig, CameraFileType
from .allocator import Allocator, ConfigAllocator

from modlib.models import COLOR_FORMAT, Model
from modlib.models.zoo import InputTensorOnly

try:
    # Runtime Arena SDK libraries available in wheel
    from modlib.devices.triton import triton_cpp
except:
    # Development environment load Arena SDK from source
    from .arena_path_resolver import load_arena_sdk

    load_arena_sdk()
    from modlib.devices.triton import triton_cpp  # noqa: E402


SENSOR_WIDTH = 4052
SENSOR_HEIGHT = 3036

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s]  %(levelname)s %(message)s")
logger = logging.getLogger(__name__.split(".")[-1])


class RequestHandler:
    def __init__(self, rps):
        self._requests = multiprocessing.Manager().list()
        self._requestslock = multiprocessing.Lock()
        self.notifyme_r, self.notifyme_w = multiprocessing.Pipe(duplex=False)

        self.rps = rps
        self._rps = Rate()

    def py_callback(self, req_idx: int):
        with self._requestslock:
            self._requests += [req_idx]

        self._rps.update()
        self.rps.value = self._rps.value

        self.notifyme_w.send(b"\x00")


class Triton(Device):
    """
    Triton® Smart Camera Model with Sony IMX501.

    This camera device module allows to run model inference on the IMX501 vision sensor.
    Output tensors are post-processed by the model post-processor function and attached to the frame.
    """

    def __init__(
        self,
        headless: Optional[bool] = False,
        enable_input_tensor: Optional[bool] = False,
        timeout: Optional[int] = None,
        frame_rate: Optional[float] = 30.0,
        image_size: Tuple[int, int] = (640, 480),
    ):
        """
        Initialize the Triton® device.

        Args:
            headless: Initialising the Triton® in headless mode means `frame.image` is never processed and unavailable.
            enable_input_tensor: When enabling input tensor, `frame.image` will be replaced by the input tensor image.
            timeout: If set, automatically stop the device loop after the specified seconds.
            frame_rate: The frames per second applied to the Arena SDK configuration.
            image_size: Resolution of the frame.image. Defaults to (640, 480) which has the original aspect ratio.
        """
        if not triton_cpp.arena_sdk_found():
            raise ImportError("Modlib was compiled without the Arena SDK")

        self.model = None
        self.roi_it = None  # ROI input tensor
        self.roi_hires = None  # ROI high resolution image

        # Shared memory for request pool & config
        self._allocator = Allocator()
        self.config = ConfigAllocator()
        self.config.allocate()

        # Frame buffer
        self._frames = []
        self._frameslock = threading.Lock()
        self._frame_ready = threading.Condition(self._frameslock)
        self.last_detections = None
        self.input_tensor_image = None

        # Process thread
        self.lock = threading.Lock()
        self._proc_started = threading.Event()
        self._proc_abort = threading.Event()
        self._proc_thread = None

        atexit.register(self.stop)
        self._triton_cpp_process = None

        self.fps = Rate()
        self.dps = Rate()
        self.rps = multiprocessing.Value("d", 0.0)

        self._rh = RequestHandler(self.rps)

        self._monitor_abort = threading.Event()
        self._monitor_rate_thread = None

        self.frame_rate = frame_rate
        self.image_size = image_size
        super().__init__(
            headless=headless,
            enable_input_tensor=enable_input_tensor,
            timeout=timeout,
        )

    def _initiate(self, headless: bool, enable_input_tensor: bool, frame_rate: float, roi_it: ROI, roi_hires: ROI):
        # NOTE: Derive how to calculate the image size from the binning factor
        # These are the perceived images when testing
        image_shape_map = {
            1: (SENSOR_WIDTH, SENSOR_HEIGHT, 3),  # RPS: 4+
            2: (2024, 1516, 3),  # RPS: 15+
            3: (1348, 1008, 3),  # RPS: 23+
            4: (1012, 756, 3),  # RPS: ~30
            5: (808, 604, 3),  # RPS: 30+
            6: (672, 504, 3),  # RPS: 30+
            7: (576, 432, 3),  # RPS: 30+
            8: (504, 378, 3),  # RPS: 30+
        }

        # find binning factor that matches the image size
        def score_mode(binning_factor, requested_size):
            actual_size = image_shape_map[binning_factor][:2]
            ar = requested_size[0] / requested_size[1]
            actual_ar = actual_size[0] / actual_size[1]

            def score_format(desired, actual):
                score = desired - actual
                return -score / 4 if score < 0 else score * 2

            score = score_format(requested_size[0], actual_size[0])
            score += score_format(requested_size[1], actual_size[1])
            score += 1500 * score_format(ar, actual_ar)
            return score

        best_binning_factor = min(image_shape_map, key=lambda bf: score_mode(bf, self.image_size))

        # Convert ROI to absolute values
        # NOTE: Triton expects input tensor ROI scaled to the sensor resolution and divisible by 4
        roi_it_abs = (
            int(roi_it[0] * SENSOR_WIDTH) // 4 * 4,
            int(roi_it[1] * SENSOR_HEIGHT) // 4 * 4,
            int(roi_it[2] * SENSOR_WIDTH) // 4 * 4,
            int(roi_it[3] * SENSOR_HEIGHT) // 4 * 4,
        )

        image_shape = image_shape_map[best_binning_factor]
        hires_w = int(roi_hires[2] * image_shape[0]) // 4 * 4
        hires_h = int(roi_hires[3] * image_shape[1]) // 4 * 4
        roi_hires_abs = (
            int(roi_hires[0] * image_shape[0]) // 4 * 4,
            int(roi_hires[1] * image_shape[1]) // 4 * 4,
            hires_w,
            hires_h,
        )

        # Initialize shared memory
        self._allocator.allocate(
            image_shape=(0, 0, 0) if self.headless or self.enable_input_tensor else (hires_w, hires_h, 3),
            input_tensor_shape=self.model.input_tensor_shape if self.enable_input_tensor else (0, 0, 0),
            output_tensor_shape_list=self.model.output_tensor_shape_list,
        )

        # Settings
        triton_config = TritonConfig(
            keep_running=False,
            headless=headless,
            enable_input_tensor=enable_input_tensor,
            binning_factor=ctypes.c_uint8(best_binning_factor),
            frame_rate=ctypes.c_double(frame_rate),
            roi_it=(ctypes.c_uint32 * 4)(*roi_it_abs),
            roi_hires=(ctypes.c_uint32 * 4)(*roi_hires_abs),
            total_pool_size=ctypes.c_uint64(self._allocator.total_size),
        )

        self.config.set_config(triton_config)

    def _process_requests_func(self):
        self._proc_started.set()

        while not self._proc_abort.is_set():
            if self._rh.notifyme_r.poll(0.2):
                self._rh.notifyme_r.recv()
                self._process_requests()

    def _process_requests(self):
        with self._rh._requestslock:
            requests = list(self._rh._requests)
            self._rh._requests[:] = []  # Clear the ListProxy object

        with self.lock:
            if not requests:
                return
            req_idx = requests.pop(0)

            # Process information in the triton request
            # Add the processed information as a Frame to the frame buffer
            self._parse_request(req_idx)

            # Release the request
            self._allocator.release_request(req_idx)

            # Release possible remaining requests, those leftovers are thrown away
            for req_idx in requests:
                logger.debug("Processing thread is dropping incoming triton requests.")
                self._allocator.release_request(req_idx)

    def _get_output_tensor_shape(self, req):
        output_tensor = req.output_tensor
        if not output_tensor:
            return []

        return [list(t.size)[: t.num_dimensions] for t in output_tensor.info[: output_tensor.num_tensors]]

    def _update_input_tensor_image(self, req):
        # TODO: check if input tensor is available
        # input_tensor = req.input_tensor
        # if not input_tensor:
        #     raise ValueError(
        #         """
        #         The provided model was converted with input tensor disabled,
        #         Provide a model with input tensor enabled.
        #     """
        #     )

        h = req.input_tensor.height
        w = req.input_tensor.width
        c = req.input_tensor.num_channels

        r1 = (
            np.frombuffer(
                self._allocator.mmap(),
                offset=req.input_tensor.data_offset,
                count=req.input_tensor.data_size // np.dtype(np.uint8).itemsize,
                dtype=np.uint8,
            )
            .reshape((c, h, w))
            .astype(np.int32)[(2, 1, 0), :, :]
        )

        norm_val = self.model.info["input_tensor"]["norm_val"]
        norm_shift = self.model.info["input_tensor"]["norm_shift"]
        div_val = self.model.info["input_tensor"]["div_val"]
        div_shift = self.model.info["input_tensor"]["div_shift"]
        for i in [0, 1, 2]:
            r1[i] = ((((r1[i] << norm_shift[i]) - norm_val[i]) << div_shift) // div_val[i]) & 0xFF

        self.input_tensor_image = np.transpose(r1, (1, 2, 0)).astype(np.uint8).copy()

    def _parse_request(self, req_idx):
        req = self._allocator.get_request(req_idx)

        detections = None
        new_detection = False

        # 1. Output Tensor processing
        if isinstance(self.model, InputTensorOnly):
            pass
        elif req.output_tensor.data_size > 0:
            np_output = np.frombuffer(
                self._allocator.mmap(),
                offset=req.output_tensor.data_offset,
                count=req.output_tensor.data_size // np.dtype(np.float32).itemsize,
                dtype=np.float32,
            ).copy()

            offset = 0
            outputs = []
            for tensor_shape in self._get_output_tensor_shape(req):
                size = np.prod(tensor_shape)
                outputs.append(np_output[offset : offset + size].reshape(tensor_shape, order="F"))
                offset += size

            # Post processing
            detections = self.model.post_process(outputs)

            self.last_detections = detections
            new_detection = True
            self.dps.update()
        elif self.last_detections is None:
            # Missing output tensor in frame & no detection yet
            # Skip adding to frame_queue
            return
        else:
            detections = self.last_detections

        # 2. High res Image or input tensor processing
        if self.enable_input_tensor:
            if req.input_tensor.data_size > 0:
                if new_detection:
                    self._update_input_tensor_image(req)
            elif self.input_tensor_image is None:
                # No input tensor in frame & no input tensor image yet
                # Skip adding to frame_queue
                return
            image = self.input_tensor_image
        else:
            rh, rw, rc = req.image.height, req.image.width, req.image.num_channels
            image = cv2.resize(
                np.frombuffer(
                    self._allocator.mmap(),
                    offset=req.image.data_offset,
                    count=req.image.data_size // np.dtype(np.uint8).itemsize,
                    dtype=np.uint8,
                ).reshape((rh, rw, rc)),  # copy required: the cv2 resize creates a copy
                self.image_size,
            )
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
                    color_format=COLOR_FORMAT.BGR,
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
                display_aspect = (SENSOR_WIDTH * W) / (SENSOR_HEIGHT * H)
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
        if self.model is None:
            raise RuntimeError(
                "Triton requires to have a model deployed for now. Until the Input tensor only model is supported."
            )
            # TODO: deploy input tensor only model
            # self.deploy(InputTensorOnly())

        # Setup & Verify ROI (input tensor & high res image)
        self._initiate_roi()
        self._verify_roi()

        self._initiate(
            headless=self.headless,
            enable_input_tensor=self.enable_input_tensor,
            frame_rate=self.frame_rate,
            roi_it=self.roi_it,
            roi_hires=self.roi_hires,
        )

        # Start processing thread
        self._proc_started.clear()
        self._proc_abort.clear()
        self._proc_thread = threading.Thread(target=self._process_requests_func)
        self._proc_thread.daemon = True
        self._proc_thread.start()
        self._proc_started.wait()

        # Call triton process start
        self._triton_cpp_process = multiprocessing.Process(
            target=triton_cpp.start,
            args=(self._rh.py_callback,),  # Pass the callback function
        )
        self._triton_cpp_process.daemon = True
        self.config.set_keep_running(True)
        self.dps.init()

        # Ignore SIGINT in the child process
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._triton_cpp_process.start()
        signal.signal(signal.SIGINT, sigint_handler)  # Restore the interupt handler

        # Warn when difference between DPS and RPS is large
        self._monitor_abort.clear()
        self._monitor_rate_thread = threading.Thread(target=self._monitor_dps_performance)
        self._monitor_rate_thread.setDaemon(True)
        self._monitor_rate_thread.start()

        logger.info("Triton started!")

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
        atexit.unregister(self.stop)

        # Stop the processing thread
        if self._proc_thread and self._proc_thread.is_alive():
            self._proc_abort.set()
            self._proc_thread.join()

        if self._monitor_rate_thread and self._monitor_rate_thread.is_alive():
            self._monitor_abort.set()
            self._monitor_rate_thread.join()

        # Stop the triton process
        if self.config._running:
            self.config.set_keep_running(False)

            if self._triton_cpp_process and self._triton_cpp_process.is_alive():
                self._triton_cpp_process.join()
                # NOTE: consider a force kill after a timeout in join

            # Release any remaining requests
            with self._rh._requestslock:
                unseen_requests = self._rh._requests
                self._rh._requests[:] = []

            for req_idx in unseen_requests:
                self._allocator.release_request(req_idx)

        self._rh.notifyme_r.close()
        self._rh.notifyme_w.close()

        self.model = None
        self.roi_it = None
        self.roi_hires = None
        self.frame_rate = None
        self.image_size = None

        # Clear any remaining frames in the framebuffer
        with self._frameslock:
            self._frames.clear()

        # Cleanup shared memory
        self._allocator.cleanup()
        self.config.cleanup()

        logger.info("Triton closed successfully.")

    def set_input_tensor_cropping(self, roi: Union[ROI, Tuple[float, float, float, float]]):
        """
        Set the input tensor cropping. Can only be adjusted during initialisation.

        Args:
            roi: The relative ROI (region of interest) in the form a (left, top, width, height) [%] crop for
                the input inference.
        """
        if self.config._running:
            raise RuntimeError("Cannot adapt input tensor cropping while running.")

        if not isinstance(roi, (Tuple, ROI)) or len(roi) != 4:
            raise ValueError("roi must be a tuple of 4 floats or the named tuple ROI.")
        if isinstance(roi, Tuple):
            roi = ROI(*roi)

        if not self.model:
            raise ValueError("No model deployed. Make sure to deploy a model before setting the input tensor cropping.")
        if not all(0 <= value <= 1 for value in roi):
            raise ValueError("All relative ROI values (left, top, width, height) must be between 0 and 1.")

        (left, top, width, height) = roi
        if left + width > 1 or top + height > 1:
            raise ValueError("ROI is out of the frame. Ensure that left + width <= 1 and top + height <= 1.")

        self.roi_it = roi

    def set_image_cropping(self, roi: Union[ROI, Tuple[float, float, float, float]]):
        """
        Set the cropping of the high resolution image. Can only be adjusted during initialisation.

        Args:
            roi: The relative ROI (region of interest) in the form a (left, top, width, height) [%] crop.
        """
        if self.config._running:
            raise RuntimeError("Cannot adapt the camera high res image cropping while running.")

        if not isinstance(roi, (Tuple, ROI)) or len(roi) != 4:
            raise ValueError("roi must be a tuple of 4 floats or the named tuple ROI.")
        if isinstance(roi, Tuple):
            roi = ROI(*roi)

        (left, top, width, height) = roi
        if not all(0 <= value <= 1 for value in roi):
            raise ValueError("All ROI values (left, top, width, height) must be between 0 and 1.")
        if left + width > 1 or top + height > 1:
            raise ValueError("ROI is out of the frame. Ensure that left + width <= 1 and top + height <= 1.")

        self.roi_hires = roi

    # def prepare_model_for_deployment(self, model: Model, overwrite: Optional[bool] = None) -> str | None:
    #     # packaged model - done
    #     if model.model_type == MODEL_TYPE.RPK_PACKAGED:
    #         network_file = model.model_file

    #     # TODO

    #     else:
    #         network_file = None

    #     # TODO

    #     return network_file

    def deploy(self, model: Model, overwrite: Optional[bool] = None) -> None:
        """
        This method manages the process to run a model on the device. This requires the
        following files/variables to be available:

        - `network.fpk` file available at `model.network_file_path`
        - `fpk_info.dat` file available at `model.info_file_path`
        - `input_tensor_shape` variable available at `model.input_tensor_shape`
        - `output_tensor_shape_list` variable available at `model.output_tensor_shape_list`

        Args:
            model: The model to be uploaded to device.
            overwrite: Unused for now (TODO).

        Raises:
            FileNotFoundError: If the packaged network file cannot be found.
        """
        # TODO: temporary extra requied triton information in model
        if not hasattr(model, "network_file_path"):
            raise AttributeError(
                "Model must have 'network_file_path' attribute for now. E.g. `model.network_file_path = ./path/to/network.fpk`"
            )
        if not hasattr(model, "info_file_path"):
            raise AttributeError(
                "Model must have 'info_file_path' attribute for now. E.g. `model.info_file_path = ./path/to/fpk_info.dat`"
            )
        if not hasattr(model, "input_tensor_shape"):
            raise AttributeError(
                "Model must have 'input_tensor_shape' attribute for now. E.g. `model.input_tensor_shape = (320, 320, 3)`"
            )
        if not hasattr(model, "output_tensor_shape_list"):
            raise AttributeError(
                "Model must have 'output_tensor_shape_list' attribute for now. E.g. `model.output_tensor_shape_list = [(100, 4), (100,), (100,), (1,)]`"
            )

        # TODO sort out Triton packager
        # Prepare model
        # network_file = self.prepare_model_for_deployment(model, overwrite)
        # if network_file is None:
        #     raise FileNotFoundError("Packaged network file error")

        # Configure model deployment
        self.model = model

        if not os.path.exists(model.network_file_path):
            raise FileNotFoundError(f"Network file not found: {model.network_file_path}")
        if not os.path.exists(model.info_file_path):
            raise FileNotFoundError(f"Info file not found: {model.info_file_path}")

        # Uploading the info file is also extracting relevant model information from the fpk_info struct
        self.model.info = triton_cpp.upload_file(
            model.info_file_path, CameraFileType.FILE_DEEP_NEURAL_NETWORK_INFO.value
        )
        triton_cpp.upload_file(model.network_file_path, CameraFileType.FILE_DEEP_NEURAL_NETWORK_NETWORK.value)

    def upload_firmware(self, firmware_file_path: str, loader_file_path: str):
        """
        Utility function to upload the firmware and loader files to the device.

        Args:
            firmware_file_path: Path to the firmware file.
            loader_file_path: Path to the loader file.

        Raises:
            FileNotFoundError: If the firmware or loader file cannot be found.
        """
        if not os.path.exists(firmware_file_path):
            raise FileNotFoundError(f"Firmware file not found: {firmware_file_path}")
        if not os.path.exists(loader_file_path):
            raise FileNotFoundError(f"Loader file not found: {loader_file_path}")

        triton_cpp.upload_file(firmware_file_path, CameraFileType.FILE_DEEP_NEURAL_NETWORK_FIRMWARE.value)
        triton_cpp.upload_file(loader_file_path, CameraFileType.FILE_DEEP_NEURAL_NETWORK_LOADER.value)

    def __enter__(self):
        """
        Start the Triton device stream.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the Triton device stream.
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
            frame = self._frames.pop(-1)
            if self._frames:
                logger.debug(f"Main thread is dropping {len(self._frames)} frames.")
                self._frames.clear()

        self.fps.update()
        return frame

    def __next__(self) -> Frame:
        """
        Get the next frame in the device stream.

        Returns:
            The next frame in the device stream.
        """
        self.check_timeout()
        return self.get_frame()

    def test_connection(self):
        """
        Utility function to test the connection to the device.
        """
        triton_cpp.test_connection()
