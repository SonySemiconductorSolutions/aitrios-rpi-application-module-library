#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm> 
#include <queue>
#include <unordered_map>
#include <utility>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "personlab_decoder.h"

// Windows does not define M_PI by default
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const size_t INPUT_TENSOR_HEIGHT = 353;
const size_t INPUT_TENSOR_WIDTH = 481;


void bilinear_interpolate(
    const float* input, int old_height, int old_width, int num_channels,
    int new_height, int new_width, float* output
) {
    float x_ratio = static_cast<float>(old_width) / new_width;
    float y_ratio = static_cast<float>(old_height) / new_height;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int y = 0; y < new_height; ++y) {
        float src_y = std::max((y + 0.5f) * y_ratio - 0.5f, 0.0f);
        int y0 = static_cast<int>(std::floor(src_y));
        int y1 = std::min(y0 + 1, old_height - 1);
        float dy = src_y - y0;

        for (int x = 0; x < new_width; ++x) {
            float src_x = std::max((x + 0.5f) * x_ratio - 0.5f, 0.0f);
            int x0 = static_cast<int>(std::floor(src_x));
            int x1 = std::min(x0 + 1, old_width - 1);
            float dx = src_x - x0;

            for (int c = 0; c < num_channels; ++c) {
                int idx00 = (y0 * old_width + x0) * num_channels + c;
                int idx10 = (y0 * old_width + x1) * num_channels + c;
                int idx01 = (y1 * old_width + x0) * num_channels + c;
                int idx11 = (y1 * old_width + x1) * num_channels + c;

                // Use a temporary accumulator
                float value = (1.0f - dx) * (1.0f - dy) * input[idx00] +
                              dx * (1.0f - dy) * input[idx10] +
                              (1.0f - dx) * dy * input[idx01] +
                              dx * dy * input[idx11];

                output[(y * new_width + x) * num_channels + c] = value;
            }
        }
    }
}


float* resize_tensors(const float* t, int old_height, int old_width, int num_channels, int new_height, int new_width) {
    float* resized_tensor = new float[new_height * new_width * num_channels];

    bilinear_interpolate(t, old_height, old_width, num_channels, new_height, new_width, resized_tensor);
    
    return resized_tensor;
}


// split_and_refine_mid_offsets
// A modified bilinear sampler that operates on a channel slice.
void bilinear_sampler_slice(
    const float* x,
    float* v,
    int x_channel_start,
    int v_channel_start,
    int H_x, int W_x, int x_channels,
    int H_v, int W_v, int v_channels
) {
    // Parallelize over both spatial dimensions of v.
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < H_v; ++i) {
        for (int j = 0; j < W_v; ++j) {
            // Pre-compute base index for v
            const int v_base = (i * W_v + j) * v_channels;
            float* const v_ptr = v + v_base;
            
            // Load values from v
            const float vx = v_ptr[v_channel_start];
            const float vy = v_ptr[v_channel_start + 1];

            // Compute sampling indices
            const float vx_floor = std::floor(vx);
            const float vy_floor = std::floor(vy);
            const int ix0 = j + static_cast<int>(vx_floor);
            const int iy0 = i + static_cast<int>(vy_floor);
            const int ix1 = ix0 + 1;
            const int iy1 = iy0 + 1;

            // Early bounds check (padding)
            if (ix0 >= 0 && iy0 >= 0 && ix1 < W_x && iy1 < H_x) {
                // Pre-compute indices for x
                const int idx00 = ((iy0 * W_x + ix0) * x_channels) + x_channel_start;
                const int idx01 = ((iy1 * W_x + ix0) * x_channels) + x_channel_start;
                const int idx10 = ((iy0 * W_x + ix1) * x_channels) + x_channel_start;
                const int idx11 = ((iy1 * W_x + ix1) * x_channels) + x_channel_start;

                // Pre-compute interpolation weights
                const float dx = vx - vx_floor;
                const float dy = vy - vy_floor;

                const float w00 = (1.0f - dx) * (1.0f - dy);
                const float w01 = (1.0f - dx) * dy;
                const float w10 = dx * (1.0f - dy);
                const float w11 = dx * dy;

                // Compute interpolated values
                const float* const x_ptr = x;
                const float out0 = w00 * x_ptr[idx00] + w01 * x_ptr[idx01] + 
                                 w10 * x_ptr[idx10] + w11 * x_ptr[idx11];
                const float out1 = w00 * x_ptr[idx00 + 1] + w01 * x_ptr[idx01 + 1] + 
                                 w10 * x_ptr[idx10 + 1] + w11 * x_ptr[idx11 + 1];

                // Update v values
                v_ptr[v_channel_start] += out0;
                v_ptr[v_channel_start + 1] += out1;
            }
        }
    }
}


void refine_mid_offsets(
    float* mid_offsets,
    const float* short_offsets,
    const std::vector<std::pair<int, int>>& edges,
    int H_x, int W_x, int x_channels,   // dimensions for short_offsets
    int H_v, int W_v, int v_channels    // dimensions for mid_offsets
) {
    // Create reversed edges.
    std::vector<std::pair<int, int>> rev_edges(edges.size());
    std::transform(edges.begin(), edges.end(), rev_edges.begin(),
                   [](const std::pair<int, int>& edge) {
                       return std::make_pair(edge.second, edge.first);
                   });
    // Concatenate the original edges and reversed edges.
    std::vector<std::pair<int, int>> edges_loop = edges;
    edges_loop.insert(edges_loop.end(), rev_edges.begin(), rev_edges.end());
    
    const int num_steps = 2;
    for (size_t mid_idx = 0; mid_idx < edges_loop.size(); ++mid_idx) {
        int offsets_channel_start = edges_loop[mid_idx].second * 2;
        int base_channel_start = mid_idx * 2;
        for (int step = 0; step < num_steps; ++step) {
            bilinear_sampler_slice(
                short_offsets, mid_offsets,
                offsets_channel_start, base_channel_start,
                H_x, W_x, x_channels,
                H_v, W_v, v_channels
            );
        }
    }
}

// Compute heatmaps
void accumulate_votes(
    const float* votes, int num_votes, 
    int H, int W, float* heatmap
) {
    // Initialize heatmap to zero
    std::fill(heatmap, heatmap + H * W, 0.0f);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < num_votes; ++i) {
        float x = votes[i * 3];
        float y = votes[i * 3 + 1];
        float p = votes[i * 3 + 2];

        int tl_x = static_cast<int>(std::floor(x));
        int tl_y = static_cast<int>(std::floor(y));
        int tr_x = static_cast<int>(std::ceil(x));
        int tr_y = tl_y;
        int bl_x = tl_x;
        int bl_y = static_cast<int>(std::ceil(y));
        int br_x = tr_x;
        int br_y = bl_y;

        float dx = x - tl_x;
        float dy = y - tl_y;

        float tl_val = p * (1.0f - dx) * (1.0f - dy);
        float tr_val = p * dx * (1.0f - dy);
        float bl_val = p * dy * (1.0f - dx);
        float br_val = p * dy * dx;

        if (tl_x >= 0 && tl_x < W && tl_y >= 0 && tl_y < H) heatmap[tl_y * W + tl_x] += tl_val;
        if (tr_x >= 0 && tr_x < W && tr_y >= 0 && tr_y < H) heatmap[tr_y * W + tr_x] += tr_val;
        if (bl_x >= 0 && bl_x < W && bl_y >= 0 && bl_y < H) heatmap[bl_y * W + bl_x] += bl_val;
        if (br_x >= 0 && br_x < W && br_y >= 0 && br_y < H) heatmap[br_y * W + br_x] += br_val;
    }
}


void compute_heatmaps(
    const float* kp_maps, const float* short_offsets, 
    int H, int W, int num_kp, float kp_radius, float* heatmaps
) {
    
    float* r = new float[H * W];
    float* votes = new float[H * W * 3];

    for (int i = 0; i < num_kp; ++i) {
        int vote_index = 0;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float this_kp_map = kp_maps[(h * W + w) * num_kp + i];
                float vote_x = w + short_offsets[(h * W + w) * num_kp * 2 + 2 * i];
                float vote_y = h + short_offsets[(h * W + w) * num_kp * 2 + 2 * i + 1];

                // Store the vote
                votes[vote_index++] = vote_x;
                votes[vote_index++] = vote_y;
                votes[vote_index++] = this_kp_map;
            }
        }

        accumulate_votes(votes, vote_index / 3, H, W, r);

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                heatmaps[(h * W + w) * num_kp + i] = r[h * W + w] / (M_PI * kp_radius * kp_radius);
            }
        }
    }

    delete[] votes;
    delete[] r;
}


void gaussian_kernel(float sigma, float* kernel, int& size, float truncate = 4.0f) {
    int radius = static_cast<int>(truncate * sigma + 0.5f);
    size = 2 * radius + 1;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        kernel[i + radius] = std::exp(-0.5f * (i / sigma) * (i / sigma));
        sum += kernel[i + radius];
    }
    // Normalize the kernel
    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }
}

auto symmetric_index = [](int idx, int n) {
    if (n <= 0) return 0;
    idx = idx % (2 * n);
    if (idx < 0) idx += 2 * n;
    return (idx < n) ? idx : 2 * n - idx - 1;
};

template <bool Horizontal>
void convolve2D(
    float* array, int H, int W, int num_kp, 
    const float* kernel, int kernel_size, 
    int channel_index
) {
    int pad = kernel_size / 2;

    // Pre compute index mapping
    int* index_map = new int[Horizontal ? W * kernel_size : H * kernel_size];
    for (int k = 0; k < kernel_size; ++k) {
        int offset = k - pad;
        if constexpr (Horizontal) {
            for (int j = 0; j < W; ++j) {
                index_map[k * W + j] = symmetric_index(j + offset, W);
            }
        } else {
            for (int i = 0; i < H; ++i) {
                index_map[k * H + i] = symmetric_index(i + offset, H);
            }
        }
    }

    // Store temporary result for cache efficiency
    float* temp = new float[H * W]();

    // Parallelize the outer loop using OpenMP
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < H; ++i) {
        for (int k = 0; k < kernel_size; ++k) {
            for (int j = 0; j < W; ++j) {
                float acc = 0.0f;
                if constexpr (Horizontal) {
                    int j_idx = index_map[k * W + j];
                    acc += array[(i * W + j_idx) * num_kp + channel_index] * kernel[k];
                } else {
                    int i_idx = index_map[k * H + i];
                    acc += array[(i_idx * W + j) * num_kp + channel_index] * kernel[k];
                }
                temp[i * W + j] += acc;
            }
        }
    }

    // Write result
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            array[(i * W + j) * num_kp + channel_index] = temp[i * W + j];
        }
    }

    delete[] temp;
    delete[] index_map;
}

// Gaussian filter
void gaussian_filter(float* image, int H, int W, int num_kp, float sigma, int channel_index, float truncate = 4.0f) {
    int kernel_size;
    float* kernel = new float[static_cast<int>(truncate * sigma * 2 + 1)];
    gaussian_kernel(sigma, kernel, kernel_size, truncate);
    convolve2D<true>(image, H, W, num_kp, kernel, kernel_size, channel_index);  // Horizontal pass
    convolve2D<false>(image, H, W, num_kp, kernel, kernel_size, channel_index); // Vertical pass
    delete[] kernel;
}

void maximum_filter(
    const float* heatmaps, int H, int W, int num_kp, 
    const std::vector<std::vector<int>>& footprint, 
    int channel_index, float* output
) {
    int footprint_height = footprint.size();
    int footprint_width = footprint[0].size();
    int pad_height = footprint_height / 2;
    int pad_width = footprint_width / 2;

    // Precompute index map
    int* index_map_i = new int[footprint_height * H];
    int* index_map_j = new int[footprint_width * W];
    for (int di = -pad_height; di <= pad_height; ++di) {
        for (int i = 0; i < H; ++i) {
            index_map_i[(di + pad_height) * H + i] = symmetric_index(i + di, H);
        }
    }
    for (int dj = -pad_width; dj <= pad_width; ++dj) {
        for (int j = 0; j < W; ++j) {
            index_map_j[(dj + pad_width) * W + j] = symmetric_index(j + dj, W);
        }
    }

    // Initialize output to zero
    std::fill(output, output + H * W, 0.0f);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int di = -pad_height; di <= pad_height; ++di) {
        for (int dj = -pad_width; dj <= pad_width; ++dj) {
            if (footprint[di + pad_height][dj + pad_width]) {
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < W; ++j) {
                        int ni = index_map_i[(di + pad_height) * H + i];
                        int nj = index_map_j[(dj + pad_width) * W + j];
                        output[i * W + j] = std::max(output[i * W + j], heatmaps[(ni * W + nj) * num_kp + channel_index]);
                    }
                }
            }
        }
    }

    delete[] index_map_i;
    delete[] index_map_j;
}


struct Keypoint {
    int id;
    std::array<float, 2> xy;
    float conf;
};


std::vector<Keypoint> get_keypoints(const float* heatmaps, int H, int W, int num_kp, const float peak_thresh) {
    std::vector<Keypoint> keypoints;
    std::vector<std::vector<int>> footprint = {
        {0, 1, 0},
        {1, 1, 1},
        {0, 1, 0}
    };

    float* peaks = new float[H * W];

    for (int i = 0; i < num_kp; ++i) {
        maximum_filter(heatmaps, H, W, num_kp, footprint, i, peaks);

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                if (peaks[h * W + w] == heatmaps[(h * W + w) * num_kp + i] && heatmaps[(h * W + w) * num_kp + i] > peak_thresh) {
                    Keypoint kp;
                    kp.id = i;
                    kp.xy = {static_cast<float>(w), static_cast<float>(h)};
                    kp.conf = heatmaps[(h * W + w) * num_kp + i];
                    keypoints.push_back(kp);
                }
            }
        }
    }

    delete[] peaks;
    return keypoints;
}


// Group keypoints
std::vector<std::pair<int, int>> iterative_bfs(
    const std::unordered_map<int, std::vector<int>>& graph,
    int start,
    const int num_keypoints
) {
    std::vector<std::pair<int, int>> path;
    std::queue<std::pair<int, int>> q;
    std::vector<bool> visited(num_keypoints, false);

    q.push({-1, start});
    while (!q.empty()) {
        auto v = q.front();
        q.pop();
        if (!visited[v.second]) {
            visited[v.second] = true;
            path.push_back(v);
            auto it = graph.find(v.second);
            if (it != graph.end()) {
                for (int w : it->second) {
                    q.push({v.second, w});
                }
            }
        }
    }
    return path;
}

void group_skeletons(
    std::vector<Keypoint>& keypoints,
    const float* mid_offsets,
    int H,
    int W,
    const int num_keypoints,
    const std::vector<std::pair<int, int>>& edges,
    const float nms_threshold,
    std::vector<std::vector<std::vector<float>>>& result
) {
    // Sort keypoints in descending order by confidence.
    std::sort(keypoints.begin(), keypoints.end(), [](const Keypoint& a, const Keypoint& b) {
        return a.conf > b.conf;
    });

    // Create directional edges: original edges + their reversals.
    std::vector<std::pair<int, int>> dir_edges = edges;
    for (const auto& edge : edges) {
        dir_edges.push_back({edge.second, edge.first});
    }

    // Build the skeleton graph.
    std::unordered_map<int, std::vector<int>> skeleton_graph;
    for (int i = 0; i < num_keypoints; ++i) {
        for (int j = 0; j < num_keypoints; ++j) {
            if (std::find(edges.begin(), edges.end(), std::make_pair(i, j)) != edges.end() ||
                std::find(edges.begin(), edges.end(), std::make_pair(j, i)) != edges.end()) {
                skeleton_graph[i].push_back(j);
                skeleton_graph[j].push_back(i);
            }
        }
    }

    // Process each keypoint.
    while (!keypoints.empty()) {
        // Pop the first keypoint (highest confidence).
        Keypoint kp = keypoints.front();
        keypoints.erase(keypoints.begin());

        // Skip this keypoint if any existing skeleton already has a keypoint
        // for this id within 10 pixels.
        bool skip = false;
        for (const auto& s : result) {
            if (std::sqrt(
                    std::pow(kp.xy[0] - s.at(kp.id)[0], 2) + 
                    std::pow(kp.xy[1] - s.at(kp.id)[1], 2)
                ) <= 10.0f
            ) {
                skip = true;
                break;
            }
        }
        if (skip) {
            continue;
        }

        // Initialize a new skeleton (num_keypoints x 3) with zeros.
        std::vector<std::vector<float>> this_skel(num_keypoints, std::vector<float>(3, 0.0f));
        // Set the root keypoint.
        this_skel[kp.id][0] = kp.xy[0];
        this_skel[kp.id][1] = kp.xy[1];
        this_skel[kp.id][2] = kp.conf;

        // Run BFS to get a path (list of edges) from the root.
        auto path = iterative_bfs(skeleton_graph, kp.id, num_keypoints);
        // Remove the first element (the root) to match the Python slicing.
        if (!path.empty())
            path.erase(path.begin());

        // Process each edge in the BFS path.
        for (const auto& edge : path) {
            // Continue only if the parent keypoint is already set.
            if (this_skel[edge.first][2] == 0.0f)
                continue;

            // Find the corresponding index (mid_idx) in the directional edges.
            auto it = std::find(dir_edges.begin(), dir_edges.end(), edge);
            if (it == dir_edges.end())
                continue;
            int mid_idx = std::distance(dir_edges.begin(), it);

            // Get the "from" keypoint coordinates, rounding to nearest integer.
            int from_kp_x = static_cast<int>(std::round(this_skel[edge.first][0]));
            int from_kp_y = static_cast<int>(std::round(this_skel[edge.first][1]));

            // Check that the rounded coordinates are within bounds.
            if (from_kp_x < 0 || from_kp_x >= W || from_kp_y < 0 || from_kp_y >= H)
                continue;

            // Calculate the number of channels in mid_offsets.
            int mid_offset_channels = 2 * 2 * edges.size();
            // Compute the index into the mid_offsets array.
            int index = (from_kp_y * W + from_kp_x) * mid_offset_channels;
            float offset_x = mid_offsets[index + 2 * mid_idx];
            float offset_y = mid_offsets[index + 2 * mid_idx + 1];

            // Compute the proposal location.
            float proposal_x = this_skel[edge.first][0] + offset_x;
            float proposal_y = this_skel[edge.first][1] + offset_y;

            // Look for candidate matches in the remaining keypoints with the expected id.
            std::vector<std::pair<int, Keypoint>> matches;
            for (size_t i = 0; i < keypoints.size(); ++i) {
                if (keypoints[i].id == edge.second) {
                    float dx = proposal_x - keypoints[i].xy[0];
                    float dy = proposal_y - keypoints[i].xy[1];
                    if (std::sqrt(dx * dx + dy * dy) <= nms_threshold) {
                        matches.push_back({static_cast<int>(i), keypoints[i]});
                    }
                }
            }

            if (matches.empty())
                continue;

            // Sort matches by their distance to the proposal.
            std::sort(matches.begin(), matches.end(), [&](const std::pair<int, Keypoint>& a,
                                                            const std::pair<int, Keypoint>& b) {
                float da = std::sqrt(std::pow(a.second.xy[0] - proposal_x, 2) +
                                     std::pow(a.second.xy[1] - proposal_y, 2));
                float db = std::sqrt(std::pow(b.second.xy[0] - proposal_x, 2) +
                                     std::pow(b.second.xy[1] - proposal_y, 2));
                return da < db;
            });

            // Choose the best match.
            int to_kp_x = static_cast<int>(std::round(matches[0].second.xy[0]));
            int to_kp_y = static_cast<int>(std::round(matches[0].second.xy[1]));
            float to_kp_conf = matches[0].second.conf;

            // Remove the matched keypoint from the list.
            keypoints.erase(keypoints.begin() + matches[0].first);

            // Update the skeleton with the matched keypoint.
            this_skel[edge.second][0] = to_kp_x;
            this_skel[edge.second][1] = to_kp_y;
            this_skel[edge.second][2] = to_kp_conf;
        }

        // Append the completed skeleton to the result.
        result.push_back(this_skel);
    }
}


void configure_num_threads() {
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    int num_threads = std::min(max_threads, 1);
    omp_set_num_threads(num_threads);
#endif
}


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
) {
    // Threaded when possible
    configure_num_threads();

    const int kp_maps_channels = num_keypoints;
    const int short_offset_channels = 2*num_keypoints;
    const int mid_offset_channels = 2*2*edges.size();
    
    // tensor post processing
    const float* resized_kp_maps = resize_tensors(kp_maps, 23, 31, kp_maps_channels, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH);
    const float* resized_shortOffset = resize_tensors(shortOffset, 23, 31, short_offset_channels, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH);
    float* resized_middleOffset = resize_tensors(middleOffset, 23, 31, mid_offset_channels, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH);

    // Split and refine mid_offsets
    refine_mid_offsets(resized_middleOffset, resized_shortOffset, edges,
        INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH, short_offset_channels,
        INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH, mid_offset_channels
    );

    // compute heatmaps
    // NOTE (performance improvement): consider using sparse matrix for heatmap since a lot of values are 0.
    float* heatmaps = new float[INPUT_TENSOR_HEIGHT * INPUT_TENSOR_WIDTH * num_keypoints];
    compute_heatmaps(resized_kp_maps, resized_shortOffset, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH, num_keypoints, kp_radius, heatmaps);
    delete[] resized_kp_maps;
    delete[] resized_shortOffset;

    // gaussian filter
    float sigma = 2.0f;
    for (int i = 0; i < num_keypoints; ++i) {
        gaussian_filter(heatmaps, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH, num_keypoints, sigma, i);
    }

    // get keypoints
    std::vector<Keypoint> keypoints = get_keypoints(heatmaps, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH, num_keypoints, peak_thresh);
    delete[] heatmaps;

    // group skeletons
    group_skeletons(keypoints, resized_middleOffset, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH, num_keypoints, edges, nms_thresh, *result);
    delete[] resized_middleOffset;
}








// #include <cstdlib>
// #include <ctime>

// int main() {
//     // Seed for random number generation
//     std::srand(static_cast<unsigned int>(std::time(nullptr)));

//     // Define dimensions
//     int H = 23;
//     int W = 31;
//     int C_kp = 2;
//     int C_short = 4;
//     int C_middle = 4;

//     // Create flat arrays for input data
//     float* kp_maps_flat = new float[H * W * C_kp];
//     float* shortOffset_flat = new float[H * W * C_short];
//     float* middleOffset_flat = new float[H * W * C_middle];

//     // Fill the flat arrays with random values between 0 and 1
//     for (int i = 0; i < H * W * C_kp; ++i) {
//         kp_maps_flat[i] = static_cast<float>(std::rand()) / RAND_MAX;
//     }
//     for (int i = 0; i < H * W * C_short; ++i) {
//         shortOffset_flat[i] = static_cast<float>(std::rand()) / RAND_MAX;
//     }
//     for (int i = 0; i < H * W * C_middle; ++i) {
//         middleOffset_flat[i] = static_cast<float>(std::rand()) / RAND_MAX;
//     }

//     int num_keypoints = 2;
//     std::vector<std::pair<int, int>> edges = {{0, 1}}; 
//     float peak_thresh = 0.004f;
//     float nms_thresh = 32.0f;

//     // Prepare the result structure
//     std::vector<std::vector<std::vector<float>>> result;

//     // Call the decode_personlab function
//     decode_personlab(kp_maps_flat, shortOffset_flat, middleOffset_flat, num_keypoints, edges, peak_thresh, nms_thresh, &result);

//     std::cout << "success" << std::endl;

//     // Clean up dynamically allocated memory
//     delete[] kp_maps_flat;
//     delete[] shortOffset_flat;
//     delete[] middleOffset_flat;

//     return 0;
// }



/// 
// g++ -o3 -o personlab_decoder personlab_decoder.cpp -I./include/ -fopenmp && ./personlab_decoder
// g++ -o3 -o personlab_decoder personlab_decoder.cpp -I./include/ && ./personlab_decoder
// valgrind --tool=callgrind ./personlab_decoder
// kcachegrind

// valgrind --leak-check=full ./personlab_decoder