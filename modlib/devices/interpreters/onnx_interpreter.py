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

from typing import Optional

import numpy as np

from modlib.devices.device import Device, Rate
from modlib.devices.frame import IMAGE_TYPE, Frame
from modlib.devices.sources import Source
from modlib.models import MODEL_TYPE, Model


class ONNXInterpreter(Device):
    """
    ONNX Interpreter device.

    This device module allows to run inference of ONNX models locally and is designed for test/development purposes.
    Input and output tensors are processed by the model pre- and post-processor functions and attached to the frame.

    Example:
    ```
    from modlib.devices import ONNXInterpreter

    device = ONNXInterpreter()
    model = ONNXModel(...)
    device.deploy(model)

    with device as stream:
        for frame in stream:
            print(frame.detections)
    ```
    """

    def __init__(
        self,
        source: Source,
        headless: Optional[bool] = False,
        timeout: Optional[int] = None,
    ):
        """
        Initialize a ONNX Interpreter device.

        Args:
            source: The source of the ONNX model.
            headless: Whether to run the interpreter in headless mode. Defaults to False.
            timeout: The timeout value for the interpreter. Defaults to None.
        """
        # NOTE: Input tensor enabled is not supported for Interpreter devices.

        self.source = source
        self.model = None
        self.onnx_model = None
        self.fps = Rate()

        super().__init__(
            headless=headless,
            enable_input_tensor=False,
            timeout=timeout,
        )

    def deploy(self, model: Model):
        """
        Deploys a ONNX model for local inference.

        Args:
            model: The ONNX model to deploy.

        Raises:
            FileNotFoundError: If the model file is not found.
            TypeError: If the model or model_file is not a ONNX model.
            AttributeError: If the model does not have a pre_process method.
        """
        if model.model_file is None:
            raise FileNotFoundError("Model file not found. Please provide a valid model file.")
        if not model.model_file.lower().endswith((".onnx")):
            raise TypeError("Model file must be a ONNX model file (.onnx)")
        if not model.model_type == MODEL_TYPE.ONNX:
            raise TypeError("Model type must be ONNX.")

        # Require model to have pre_process method
        if not hasattr(model, "pre_process"):
            raise AttributeError("Model must have a pre_process method to use KerasInterpreter.")

        self.model = model
        self.onnx_model = self.load_torch_onnx_model(model.model_file)

    def __enter__(self):
        """
        Start the ONNXInterpreter device stream.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the ONNXInterpreter device stream.
        """
        pass

    def __iter__(self):
        """
        Iterate over the frames in the device stream.
        """
        self.fps.init()
        return self

    def __next__(self) -> Frame:
        """
        Get the next frame in the device stream.

        Returns:
            The next frame in the device stream.
        """
        self.check_timeout()
        self.fps.update()

        input_frame = self.source.get_frame()
        if input_frame is None:
            raise StopIteration

        if self.model:
            # NOTE: It might be possible to avoid forcing model.pre_process to return input_tensor_image
            # But one needs to compensate for the padding that might happen in the proposer when visualizing
            # the detections on top of the original input_frame from source (to be investigated)

            image, input_tensor = self.model.pre_process(input_frame)
            image_type = IMAGE_TYPE.INPUT_TENSOR
            height, width, channels = image.shape

            input_name = self.onnx_model.get_inputs()[0].name
            output_names = [x.name for x in self.onnx_model.get_outputs()]
            input_tensor_bchw = np.transpose(input_tensor, (0, 3, 1, 2)).astype(np.float32)

            # Run inference
            output_tensors = self.onnx_model.run(output_names, {input_name: input_tensor_bchw})

            squeezed_tensors = [np.squeeze(t) if t.ndim > 1 else t for t in output_tensors]
            detections = self.model.post_process(squeezed_tensors)

        else:
            image = input_frame
            image_type = IMAGE_TYPE.SOURCE
            width, height, channels = self.source.width, self.source.height, self.source.channels
            detections = None

        return Frame(
            timestamp=self.source.timestamp.isoformat(),
            image=image,
            image_type=image_type,
            width=width,
            height=height,
            channels=channels,
            detections=detections,
            new_detection=True if self.model else False,
            fps=self.fps.value,
            dps=self.fps.value,
            color_format=self.source.color_format,
            input_tensor=None,
        )

    @staticmethod
    def load_torch_onnx_model(model_path: str):
        """
        Loads the onnx model file.
        Requires onnx runtime and some other external modules to be installed.

        Raises:
            ImportError: When loading the model fails due to missing dependency.
        """

        try:
            import mct_quantizers as mctq
            import onnxruntime

            # The following line is needed to make nms_ort available for onnxruntime.InferenceSession
            from sony_custom_layers.pytorch.object_detection import nms_ort  # noqa

            model = onnxruntime.InferenceSession(
                model_path, mctq.get_ort_session_options(), providers=["CPUExecutionProvider"]
            )

            return model
        except ImportError:
            raise ImportError(
                """
                onnxruntime is not installed.
                """
            )
        except Exception as e:
            raise Exception(f"Failed to load onnx model: {e}")
