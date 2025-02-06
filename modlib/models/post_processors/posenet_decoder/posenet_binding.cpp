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
#include "posenet_binding.h"
#include <algorithm>


#define N_SCORE 12121
#define N_SHORTOFFSET  24242
#define N_MIDDLEOFFSET 45632
#define MAX_DETECTIONS 10
#define STRIDES 16
#define NMS_RADIUS 10.0
#define SCORE_THRESHOLD 0.0
#define MID_SHORT_OFFSET_REFINEMENT_STEPS 5


static void TransposeTensor_imx500(float* dst_data, float* trans_dst_data, int c, int w, int h) {
    for (int i = 0; i < c; ++i) {
      for (int j = 0; j < w; ++j) {
        for (int k = 0; k < h; ++k) {
            trans_dst_data[c*w*k + c*j + i] = dst_data[w*h*i + j*h + k];
          }
        }
    }
}


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

    float score_trans[N_SCORE], shortOffset_trans[N_SHORTOFFSET], middleOffset_trans[N_MIDDLEOFFSET];
    TransposeTensor_imx500(score, score_trans, 17, 31, 23);
    TransposeTensor_imx500(shortOffset, shortOffset_trans, 34, 31, 23);
    TransposeTensor_imx500(middleOffset, middleOffset_trans, 64, 31, 23);

    result->n_detections = coral::posenet_decoder_op::DecodeAllPoses(
        score_trans, shortOffset_trans, middleOffset_trans, 23, 31,
        MAX_DETECTIONS, SCORE_THRESHOLD, MID_SHORT_OFFSET_REFINEMENT_STEPS,
        NMS_RADIUS / STRIDES, STRIDES, result->pose_keypoints, result->pose_keypoint_scores, result->pose_scores
    );

    result->n_detections = coral::posenet_decoder_op::DecodeAllPoses(
        score_trans,
        shortOffset_trans,
        middleOffset_trans,
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
