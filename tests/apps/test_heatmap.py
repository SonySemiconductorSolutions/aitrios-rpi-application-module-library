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

import pytest

from modlib.apps import Heatmap

from tests.test_devices import test_apps_device


def test_heatmap_set_methods():

    # Initialisation 
    heatmap = Heatmap(cell_size=10)
    
    assert(heatmap.cell_size == 10)
    assert(heatmap.frame_size == None)  # Default

    # Set different heatmap settings
    HEIGHT = 500
    WIDTH = 500
    CELL_SIZE = 10
    
    heatmap.set_frame_size(WIDTH, HEIGHT)
    heatmap.set_cell_size(CELL_SIZE)

    assert(heatmap.cell_size == CELL_SIZE)
    assert(heatmap.frame_size == (WIDTH, HEIGHT))


def test_heatmap_live(test_apps_device):

    heatmap = Heatmap(cell_size=50)

    with test_apps_device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.class_id == 0] # Person
            detections = detections[detections.confidence > 0.50]
            
            heatmap.update(frame, detections)
            
            # NOTE: Manually check the heatmap
            # frame.display()