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

import cv2

from modlib.devices import Triton

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from triton_models import TritonAnomaly


def main():
    device = Triton()
    model = TritonAnomaly()
    device.deploy(model)

    image_thresh = 0.45
    pixel_thresh = 0.45

    with device as stream:
        for frame in stream:
            _result = "Anomaly" if (frame.detections.score > image_thresh) else "Normal"
            mask = frame.detections.get_mask(score_threshold=pixel_thresh)
            mask = cv2.resize(mask, frame.image.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            frame.image = cv2.addWeighted(frame.image, 1, mask, 0.6, 0)
            frame.display()


if __name__ == "__main__":
    main()
