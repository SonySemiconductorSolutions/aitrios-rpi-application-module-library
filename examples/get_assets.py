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

from tests.utils import get_coco_samples, get_imagenet_samples, get_tracking_video

ASSETS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/assets"


def get_assets():

    get_imagenet_samples(os.path.join(ASSETS_DIR, "imagenet_samples"))
    get_coco_samples(os.path.join(ASSETS_DIR, "coco_samples"))
    get_tracking_video(ASSETS_DIR)


if __name__ == "__main__":
    get_assets()
