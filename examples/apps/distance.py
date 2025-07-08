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
import numpy as np

from modlib.apps import Annotator, ColorPalette
from modlib.apps.calculate import calculate_distance_matrix
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416

model = NanoDetPlus416x416()
device = AiCamera()
device.deploy(model)

annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.50]

        # Calculate distance matrix
        xc, yc = detections.center_points
        xc, yc = xc * frame.width, yc * frame.height
        dist_matrix = calculate_distance_matrix(xc, yc)

        # Display distance to each objects
        indeces = np.triu_indices(len(xc), k=1)
        for i, j in zip(*indeces):
            dist = dist_matrix[i, j]
            p1 = (int(xc[i]), int(yc[i]))
            p2 = (int(xc[j]), int(yc[j]))

            cv2.line(frame.image, p1, p2, (255, 255, 255), 1)
            cv2.putText(frame.image, f"{dist:.1f}", ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, )

        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
        frame.image = annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)

        frame.display()
