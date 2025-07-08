extern "C" {
    void decode_personlab(
        const float* kp_maps,
        const float* shortOffset,
        const float* middleOffset,
        const int num_keypoints,
        const std::vector<std::pair<int, int>>& edges,
        const float peak_thresh,
        const float nms_thresh,
        const float kp_radius,
        std::vector<std::vector<std::vector<float>>>* result
    );
}