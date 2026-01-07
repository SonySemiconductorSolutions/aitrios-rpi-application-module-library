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


class KerasInterpreter(Device):
    """
    Keras Interpreter device.

    This device module allows to run inference of Keras models locally and is designed for test/development purposes.
    Output tensors are post-processed by the model post-processor function and attached to the frame.

    Example:
    ```
    from modlib.devices import KerasInterpreter

    device = KerasInterpreter()
    model = CustomKerasModel(...)
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
        Initialize a Keras Interpreter device.

        Args:
            source: The source of the Keras model.
            headless: Whether to run the interpreter in headless mode. Defaults to False.
            timeout: The timeout value for the interpreter. Defaults to None.
        """
        # NOTE: Input tensor enabled is not supported for Interpreter devices.

        self.source = source
        self.model = None
        self.fps = Rate()

        super().__init__(
            headless=headless,
            enable_input_tensor=False,
            timeout=timeout,
        )

    def deploy(self, model: Model):
        """
        Deploys a Keras model for local inference.

        Args:
            model: The Keras model to deploy.

        Raises:
            FileNotFoundError: If the model file is not found.
            TypeError: If the model or model_file is not a Keras model.
            AttributeError: If the model does not have a pre_process method.
        """
        if model.model_file is None:
            raise FileNotFoundError("Model file not found. Please provide a valid model file.")
        if not model.model_file.lower().endswith((".keras")):
            raise TypeError("Model file must be a Keras model file (.keras)")
        if not model.model_type == MODEL_TYPE.KERAS:
            raise TypeError("Model type must be Keras.")

        # Require model to have pre_process method
        if not hasattr(model, "pre_process"):
            raise AttributeError("Model must have a pre_process method to use KerasInterpreter.")

        self.model = model
        self.keras_model = self.load_tf_keras_model(model.model_file)

    def __enter__(self):
        """
        Start the KerasInterpreter device stream.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop the KerasInterpreter device stream.
        """
        pass

    def __iter__(self):
        """
        Iterate over the frames in the device stream.
        """
        self.fps.init()
        return self

    def __next__(self):
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
            width, height, channels = image.shape

            # Inference
            output_tensors = self.keras_model.predict(input_tensor, verbose=0)

            # Post-process
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
    def load_tf_keras_model(model_path: str):
        """
        Loads the keras model file as a `tf.keras.model`.
        Requires tensorflow 2.14 to be installed.

        Raises:
            ImportError: When loading the model fails due to missing tensorflow dependency.
        """

        try:
            import tensorflow as tf

            return tf.keras.models.load_model(model_path)
        except ImportError:
            raise ImportError(
                """
                tensorflow is not installed. Please install the runtime dependencies for the selected interpreter:\n\n
                `pip install tensorflow==2.14`\n
                """
            )
