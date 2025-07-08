/*
 * Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "posenet_decoder.h"


#define N_SCORE 12121
#define N_SHORTOFFSET  24242
#define N_MIDDLEOFFSET 45632
#define MAX_DETECTIONS 10
#define STRIDES 16
#define NMS_RADIUS 10.0
#define SCORE_THRESHOLD 0.0
#define MID_SHORT_OFFSET_REFINEMENT_STEPS 5
#define NUM_KEYPOINTS 17


extern "C" {
    struct PosenetOutputDataType {
        int n_detections;
        float pose_scores[MAX_DETECTIONS];
        coral::posenet_decoder_op::PoseKeypoints pose_keypoints[MAX_DETECTIONS];
        coral::posenet_decoder_op::PoseKeypointScores pose_keypoint_scores[MAX_DETECTIONS];
    };

    void decode_poses(float* score, float* shortOffset, float* middleOffset, PosenetOutputDataType* result);
}