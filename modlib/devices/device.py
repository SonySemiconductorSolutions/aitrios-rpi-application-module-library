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

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional


from ..models import Model
from .frame import Frame


class Rate:
    """
    A class used to maintain and calculate a rolling average rate (e.g., frames per second).

    The class accumulates timing deltas between consecutive updates in a deque and provides
    the average rate based on the number of appended deltas over their sum.
    """

    def __init__(self, window: int = 30):
        """
        Initialize the Rate instance with a specified window size.

        Args:
            window: The maximum number of time deltas stored to compute the ongoing average.
        """
        self.value = 0
        self.times = deque(maxlen=window)
        self.last_time = time.perf_counter()

    def init(self):
        """
        Re-initialize timing.
        This should typically be called prior to collecting rate updates to reset the reference time.
        """
        self.last_time = time.perf_counter()

    def update(self):
        """
        Record the time elapsed since the last update, then recalculate and store the new average rate.

        This method appends the latest time delta to the deque and uses it (along with previously stored
        deltas) to compute the new rolling average.
        """
        current_time = time.perf_counter()
        self.times.append(current_time - self.last_time)
        self.value = len(self.times) / sum(self.times)
        self.last_time = current_time

    def __repr__(self) -> str:
        """
        String representation of the current average rate.

        Returns:
            The current stored rate as a string.
        """
        return str(self.value)


class Device(ABC):
    """
    Abstract base class for devices.

    A device in the Application Module Library has to conform to the following interface:
    1. Deploy a model to the device. Example usage:
    ```
    model = Model()
    device.deploy(model, *args)
    ```

    2. Enter and iterate over the frames in the device stream. Example usage:
    ```
    with device as stream:
        for frame in stream:
            ...
    ```
    """

    def __init__(
        self,
        headless: Optional[bool] = False,
        enable_input_tensor: Optional[bool] = False,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialisation of the device base class.

        Args:
            headless: Initialising in headless mode means `frame.image` is never processed and unavailable.
            enable_input_tensor: When enabling input tensor, `frame.image` will be replaced by the input tensor image.
            timeout: If set, automatically stop the device loop after the specified seconds.
        """

        self.headless = headless
        self.enable_input_tensor = enable_input_tensor
        self.timeout = timeout

        self.start_time = time.perf_counter()

    @abstractmethod
    def deploy(self, model: Model, *args):
        """
        Abstract method to deploy a model to the device.

        Args:
            model: The model to be deployed.
            *args: Additional arguments for the deployment depending on the device class.
        """
        pass

    @abstractmethod
    def __enter__(self):
        """
        Abstract method to enter a device stream.
        Assumes to set the start time for the device stream to check time-out.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Abstract method to exit the device stream.
        This method should handle cleaning up or closing the device,
        and possibly handling exceptions that occurred within the 'with' block.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Abstract method to iterate over the frames in the device stream.
        """
        pass

    @abstractmethod
    def __next__(self) -> Frame:
        """
        Abstract method to get the next frame in the device stream.

        Returns:
            The next frame in the device stream.
        """
        self.check_timeout()
        pass

    def check_timeout(self):
        """
        Utility method for checking if the specified timeout if it has been set.
        Stops the stream iterator if the timeout has been exceeded.
        """
        elapsed_time = time.perf_counter() - self.start_time
        if self.timeout is not None and elapsed_time > self.timeout:
            self.__exit__(None, None, None)
            raise StopIteration
