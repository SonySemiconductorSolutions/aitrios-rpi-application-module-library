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
#include "posenet_wrapper.h"
#include <algorithm>


void decode_poses(float* score, float* shortOffset, float* middleOffset, PosenetOutputDataType* result) {
    
    for (auto i = 0; i < N_SCORE; i++) {
        score[i] = std::clamp(score[i], -10.0f, 10.0f);
    }
    for (auto i = 0; i < N_SHORTOFFSET; i++) {
        shortOffset[i] = std::clamp(shortOffset[i], -10.0f, 10.0f) / STRIDES;
    }
    for (auto i = 0; i < N_MIDDLEOFFSET; i++) {
        middleOffset[i] = std::clamp(middleOffset[i], -162.35f, 191.49f) / STRIDES;
    }

    result->n_detections = coral::posenet_decoder_op::DecodeAllPoses(
        score, shortOffset, middleOffset, 23, 31,
        MAX_DETECTIONS, SCORE_THRESHOLD, MID_SHORT_OFFSET_REFINEMENT_STEPS,
        NMS_RADIUS / STRIDES, STRIDES, result->pose_keypoints, result->pose_keypoint_scores, result->pose_scores
    );

    result->n_detections = coral::posenet_decoder_op::DecodeAllPoses(
        score,
        shortOffset,
        middleOffset,
        23,  // height in block space
        31,  // width in block space
        MAX_DETECTIONS,
        SCORE_THRESHOLD,
        MID_SHORT_OFFSET_REFINEMENT_STEPS,
        NMS_RADIUS / STRIDES,
        STRIDES,
        result->pose_keypoints,
        result->pose_keypoint_scores,
        result->pose_scores
    );
}
