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

import ctypes

MAX_DETECTIONS = 10
NUM_KEYPOINTS = 17


class PoseKeypoints(ctypes.Structure):
    _fields_ = [("keypoints", ctypes.c_float * (NUM_KEYPOINTS * 2))]


class PoseKeypointScores(ctypes.Structure):
    _fields_ = [("scores", ctypes.c_float * NUM_KEYPOINTS)]


class PosenetOutputDataType(ctypes.Structure):
    _fields_ = [
        ("n_detections", ctypes.c_int),
        ("pose_scores", ctypes.c_float * MAX_DETECTIONS),
        ("pose_keypoints", PoseKeypoints * MAX_DETECTIONS),
        ("pose_keypoint_scores", PoseKeypointScores * MAX_DETECTIONS),
    ]
