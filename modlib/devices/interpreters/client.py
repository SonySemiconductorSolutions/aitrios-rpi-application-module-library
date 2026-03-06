#
# Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.
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

import io
import atexit
from typing import Optional, List, Union

import requests
import numpy as np

from modlib.devices.device import Device, Rate
from modlib.devices.frame import IMAGE_TYPE, Frame
from modlib.devices.sources import Source
from modlib.models import Model, Classifications, Detections, Poses, Segments, InstanceSegments, Anomaly


class InterpreterClient(Device):
    """
    Client Modlib device for a external interpreter server.

    Use it as a development device to build Modlib applications or
    to evaluate models against a running interpreter server.

    The server must expose:
    - `POST /init` to load a model and return a session id
    - `GET /input_tensor_size/<session_id>` to get the input tensor size for that model
    - `POST /infer/<session_id>` to run inference for that session
    - `DELETE /session/<session_id>` to close the session

    Please read the interpreter server documentation for more details on how to build and run it.
    """

    def __init__(
        self,
        source: Optional[Source] = None,
        endpoint: Optional[str] = None,
        headless: Optional[bool] = False,
        timeout: Optional[int] = None,
        enable_input_tensor: Optional[bool] = False,
    ):
        """
        Initialize the interpreter client device.

        Args:
            source: Optional source that yields input frames to process. Defaults to None.
            endpoint: Base URL of the interpreter server.
            headless: Disable image processing when set. Defaults to False.
            timeout: Optional timeout in seconds for the device loop. Defaults to None.
            enable_input_tensor: When enabling input tensor, `frame.image` will be replaced by the input tensor image.
        """
        super().__init__(
            headless=headless,
            enable_input_tensor=enable_input_tensor,
            timeout=timeout,
        )

        self.source = source
        self.endpoint = endpoint

        self.model = None
        self.fps = Rate()

        atexit.register(self.stop)
        self.session_id = None
        self._running = False

    def deploy(self, model: Model, data: dict):
        """
        Deploy/load a model on the remote interpreter server.
        This will send a POST request `requests.post(<endpoint>/init, json=data)` to the interpreter server.
        And expect a response with a session id in the `session_id` field if successful.

        Args:
            model: Model instance used for pre- and post-processing.
            data: Payload sent in the POST request to the `/init` endpoint.
        """
        self.model = model

        if not self.endpoint:
            raise ValueError(
                "Endpoint not set. Please specify the 'endpoint' argument when "
                "creating InterpreterClient, e.g. InterpreterClient(source, endpoint='http://localhost:8000')."
            )
        print(f"Connecting to interpreter server: {self.endpoint}")

        r = requests.post(
            f"{self.endpoint}/init",
            json=data,
        )

        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(f"Status: {r.status_code}, Reason: {r.text}")

        response = r.json()
        if response.get("status") == "error":
            raise RuntimeError(f"Model loading failed: {response.get('error', 'Unknown error')}")

        self.session_id = response["session_id"]
        self._set_network_info()
        print(f"✓ Model loaded successfully (session: {self.session_id})")

    def _set_network_info(self):
        """
        Set the network info for the model.
        This will send a GET request to the interpreter server.
        `requests.get(<endpoint>/input_tensor_size/<session_id>)`
        And expect a response with the input tensor size in the body.
        """
        r = requests.get(f"{self.endpoint}/input_tensor_size/{self.session_id}")

        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(f"Status: {r.status_code}, Reason: {r.text}")

        response = r.json()
        if response.get("status") == "error":
            raise RuntimeError(f"Failed to set network info: {response.get('error', 'Unknown error')}")

        self.model.info = {
            "input_tensor": {
                "width": response["input_tensor_size"][0],
                "height": response["input_tensor_size"][1],
            },
        }

    def __enter__(self):
        """Start the interpreter client."""
        self._running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the client."""
        self.stop()

    def __iter__(self):
        """Initialize the frame iterator."""
        if not self.source:
            raise ValueError(
                "Source not set. Please specify the 'source' argument when "
                "creating InterpreterClient, e.g. InterpreterClient(source=Video(...))."
            )

        self.fps.init()
        return self

    def __next__(self):
        """
        Fetch the next frame, run inference if a model is set, and return a frame.

        Returns:
            Frame with optional detections and performance metadata.
        """
        self.check_timeout()
        self.fps.update()

        input_frame = self.source.get_frame()
        if input_frame is None:
            raise StopIteration

        if self.model:
            # Pre-process
            it_image, it, roi = self.model.pre_process(input_frame.copy())
            if self.enable_input_tensor:
                image = it_image
                image_type = IMAGE_TYPE.INPUT_TENSOR
                w, h, c = image.shape
                color_format = self.model.color_format
                # roi = ROI(left=0, top=0, width=1, height=1) not needed?
            else:
                image = input_frame
                image_type = IMAGE_TYPE.SOURCE
                w, h, c = self.source.width, self.source.height, self.source.channels
                color_format = self.source.color_format

            # Infer
            detections = self.infer(it)
        else:
            image = input_frame
            image_type = IMAGE_TYPE.SOURCE
            w, h, c = self.source.width, self.source.height, self.source.channels
            color_format = self.source.color_format
            detections = None

        return Frame(
            timestamp=self.source.timestamp.isoformat(),
            image=image,
            image_type=image_type,
            width=w,
            height=h,
            channels=c,
            detections=detections,
            new_detection=True if self.model else False,
            fps=self.fps.value,
            dps=self.fps.value,
            color_format=color_format,
            roi=roi,
            frame_count=0,
        )

    def _infer(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on the remote interpreter server.
        This will send a POST request to the interpreter server.
        `requests.post(<endpoint>/infer/<session_id>, files={"input_npy": ("input.npy", input_tensor, "application/octet-stream")})`
        And expect a response with the output tensors in the body.

        Args:
            input_tensor: Input tensor.

        Returns:
            List of output tensors from the interpreter server.
        """
        # Ensure input_tensor has a batch dimension
        # If shape is (H, W, C), add batch dimension to make it (1, H, W, C)
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        buf = io.BytesIO()
        np.save(buf, input_tensor)
        buf.seek(0)

        r = requests.post(
            f"{self.endpoint}/infer/{self.session_id}",
            files={"input_npy": ("input.npy", buf.getvalue(), "application/octet-stream")},
        )
        r.raise_for_status()

        # Load output tensors
        npz = np.load(io.BytesIO(r.content), allow_pickle=False)
        output_tensors = [npz[k] for k in sorted(npz.files)]

        return output_tensors

    def infer(
        self, input_tensor: np.ndarray
    ) -> Union[Classifications, Detections, Poses, Segments, InstanceSegments, Anomaly]:
        """
        Run inference on the remote interpreter server.
        And return the post-processed result.

        Args:
            input_tensor: Input tensor.

        Returns:
            Post-processed result.
        """

        output_tensors = self._infer(input_tensor)
        return self.model.post_process(output_tensors)

    def stop(self):
        """
        Close the active interpreter session.
        This will send a DELETE request to the interpreter server.
        `requests.delete(<endpoint>/session/<session_id>)` and expect a 200 OK response.
        """
        if self.session_id is not None:
            # Close session
            print("\nClosing session...")
            requests.delete(f"{self.endpoint}/session/{self.session_id}")
            self.session_id = None
            print("Done.")

        self._running = False
