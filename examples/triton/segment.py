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

# BUG: device is failing to extract the tensorbuffer from the DNN Chunk data.

from modlib.apps import Annotator
from modlib.devices import Triton

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from triton_models import TritonSegmentation


def main():
    device = Triton(enable_input_tensor=True)
    model = TritonSegmentation()
    device.deploy(model)

    annotator = Annotator()

    with device as stream:
        for frame in stream:
            annotator.annotate_segments(frame, frame.detections)
            frame.display()


if __name__ == "__main__":
    main()
