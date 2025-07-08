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

#include <Python.h>
#include <cstdint>
#include <iostream>
#include <memory>

#define MAX_NUM_DIMENSIONS 16
#define MAX_NUM_TENSORS 16
#define REQUEST_POOL_SIZE 10

// -----------------------------------------------------------------------------
//  RequestInterface
// -----------------------------------------------------------------------------


struct RequestImage {
    uint32_t width;
    uint32_t height;
    uint32_t num_channels;
    uint32_t data_offset;
    uint32_t data_size;
};


struct RequestInputTensor {
    uint32_t width;
    uint32_t height;
    uint32_t num_channels;
    uint32_t data_offset;
    uint32_t data_size;
};


struct OutputTensorInfo {
    uint32_t tensor_data_num;
    uint32_t num_dimensions;
    uint16_t size[MAX_NUM_DIMENSIONS];
};


struct RequestOutputTensor {
    uint32_t num_tensors;
    OutputTensorInfo info[MAX_NUM_TENSORS];
    uint32_t data_offset;
    uint32_t data_size;
};


struct RequestInterface {
    uint8_t idx;
    RequestImage image;
    RequestInputTensor input_tensor;
    RequestOutputTensor output_tensor;   
};


struct TritonConfig {
    bool keep_running;
    bool headless;
    bool enable_input_tensor;
    uint8_t binning_factor;
    double frame_rate;
    uint32_t roi_it[4];
    uint32_t roi_hires[4];
    uint64_t total_pool_size;
};


// -----------------------------------------------------------------------------
//  RequestPool
// -----------------------------------------------------------------------------

struct RequestPool {
    bool in_use[REQUEST_POOL_SIZE];
    RequestInterface requests[REQUEST_POOL_SIZE];
    uint8_t current_index;
    void* memory_base;

    uint32_t image_buffer_offset;
    uint32_t input_tensor_offset;
    uint32_t output_tensor_offset;

    uint32_t pool_size;
    uint32_t image_size;
    uint32_t input_tensor_size;
    uint32_t output_tensor_size;

    // Initialize from pre-allocated memory
    void init() {
        current_index = 0;

        // Initialize all requests in the provided memory
        for (int i = 0; i < REQUEST_POOL_SIZE; i++) {
            // requests[i] = new (static_cast<char*>(memory_base) + pool_size + i * sizeof(RequestInterface)) RequestInterface();
            requests[i].idx = i;
            in_use[i] = false;
        }
    }

    // Cleanup without deallocating memory
    void cleanup() {
        // for (int i = 0; i < REQUEST_POOL_SIZE; i++) {
        //     if (requests[i]) {
        //         requests[i]->~RequestInterface();
        //         requests[i] = nullptr;
        //     }
        // }

        memory_base = nullptr;
    }

    RequestInterface* get_next_request() {
        // Find the next available request
        for (int i = 0; i < REQUEST_POOL_SIZE; i++) {
            int idx = (current_index + i) % REQUEST_POOL_SIZE;
            if (!in_use[idx]) {
                in_use[idx] = true;
                current_index = (idx + 1) % REQUEST_POOL_SIZE;
                return &requests[idx];
            }
        }
        return nullptr; // All requests are in use
    }

    void print_pool_usage() {
        std::string usage = "Pool usage: ";
        for (int i = 0; i < REQUEST_POOL_SIZE; i++) {
            usage += (in_use[i] ? "1" : "0");
            usage += " ";
        }
        std::cout << usage << std::endl;
    }
};


// -----------------------------------------------------------------------------
//  RequestPool helpers
// -----------------------------------------------------------------------------

uint32_t req_get_image_offset(RequestPool* rp, uint8_t idx);
uint32_t req_get_input_tensor_offset(RequestPool* rp, uint8_t idx);
uint32_t req_get_output_tensor_offset(RequestPool* rp, uint8_t idx);
uint8_t* req_get_image_ptr(RequestPool* rp, uint8_t idx);
uint8_t* req_get_input_tensor_ptr(RequestPool* rp, uint8_t idx);
float* req_get_output_tensor_ptr(RequestPool* rp, uint8_t idx);


// -----------------------------------------------------------------------------
//  Binding
// -----------------------------------------------------------------------------

struct DATNetworkInfo {
    bool is_dat_info;
    int norm_val[3];
    int norm_shift[3];
    int div_val[3];
    int div_shift;
};


extern "C" {
    void start_triton(PyObject* py_callback);
    DATNetworkInfo triton_upload_file(const char* filePath, int fileType);
    void simple_acquisition();
}