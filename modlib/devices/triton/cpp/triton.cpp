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
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
#include <map>
#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <sys/stat.h>

// for waiting for system close
#include <chrono>
#include <thread>

#include "ArenaApi.h"

#include "IMX501Utils.h"
#include "triton.h"
#include "parser.h"
#include "progress_bar.h"

#define TIMEOUT 2000
#define SENSOR_W  4052
#define SENSOR_H  3036

// -----------------------------------------------------------------------------
//  Shared TritonConfig
// -----------------------------------------------------------------------------

#ifdef _WIN32

#include <windows.h>

TritonConfig* get_shared_config() {
    HANDLE hMapFile = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        "Local\\TritonConfig"
    );

    if (hMapFile == NULL) {
        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
        return nullptr;
    }

    void* pBuf = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(TritonConfig)
    );

    if (pBuf == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return nullptr;
    }

    return reinterpret_cast<TritonConfig*>(pBuf);
}

#else // Linux

#include <fcntl.h>    // O_* constants
#include <sys/mman.h> // mmap
#include <unistd.h>   // close

TritonConfig* get_shared_config() {
    int fd = shm_open("/TritonConfig", O_RDWR, 0666);
    if (fd == -1) {
        perror("shm_open failed");
        return NULL;
    }

    void* addr = mmap(NULL, sizeof(TritonConfig), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);  // Safe to close after mmap

    if (addr == MAP_FAILED) {
        perror("mmap failed");
        return NULL;
    }

    return (TritonConfig*)addr;
}

#endif

// -----------------------------------------------------------------------------
//  RequestPool
// -----------------------------------------------------------------------------

#ifdef _WIN32

RequestPool* get_shared_request_pool(uint64_t total_pool_size) {
    HANDLE hMapFile = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        "Local\\TritonRequestPool"
    );

    if (hMapFile == NULL) {
        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
        return nullptr;
    }

    void* pBuf = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        total_pool_size
    );

    if (pBuf == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return nullptr;
    }

    RequestPool* pool = static_cast<RequestPool*>(pBuf);
    pool->memory_base = pBuf;

    return pool;
}

#else // Linux

RequestPool* get_shared_request_pool(uint64_t total_pool_size) {
    int fd = shm_open("/TritonRequestPool", O_RDWR, 0666);
    if (fd == -1) {
        perror("shm_open failed");
        return nullptr;
    }

    void* addr = mmap(NULL, total_pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (addr == MAP_FAILED) {
        perror("mmap failed");
        return nullptr;
    }

    RequestPool* pool = static_cast<RequestPool*>(addr);
    pool->memory_base = addr; // Optional: for offset calculations

    return pool;
}

#endif

// -----------------------------------------------------------------------------
//  RequestPool helpers
// -----------------------------------------------------------------------------

uint32_t req_get_image_offset(RequestPool* rp, uint8_t idx) {
    return rp->image_buffer_offset + idx * rp->image_size;
}

uint32_t req_get_input_tensor_offset(RequestPool* rp, uint8_t idx) {
    return rp->input_tensor_offset + idx * rp->input_tensor_size;
}

uint32_t req_get_output_tensor_offset(RequestPool* rp, uint8_t idx) {
    return rp->output_tensor_offset + idx * rp->output_tensor_size;
}

uint8_t* req_get_image_ptr(RequestPool* rp, uint8_t idx) {
    return static_cast<uint8_t*>(rp->memory_base) + req_get_image_offset(rp, idx);
}

uint8_t* req_get_input_tensor_ptr(RequestPool* rp, uint8_t idx) {
    return static_cast<uint8_t*>(rp->memory_base) + req_get_input_tensor_offset(rp, idx);
}

float* req_get_output_tensor_ptr(RequestPool* rp, uint8_t idx) {
    return reinterpret_cast<float*>(static_cast<uint8_t*>(rp->memory_base) + req_get_output_tensor_offset(rp, idx));
}


// -----------------------------------------------------------------------------
//  core processing
// -----------------------------------------------------------------------------
void run(RequestInterface* req, Arena::IImage* pImage, RequestPool* rp, ArenaExample::IMX501Utils& util, TritonConfig* cfg) {

    Arena::IChunkData *pChunkData = pImage->AsChunkData();
    if (pChunkData != nullptr)
    {
        if (pChunkData->IsIncomplete() == false)
        {
            
            // 1. Hires image processing (if not headless and not enable_input_tensor)            
            if (!cfg->headless && !cfg->enable_input_tensor) {
    
                // std::cout << "[INFO] processing hires image" << std::endl;

                Arena::IImage* pImage_bgr8 = Arena::ImageFactory::Convert(pImage, BGR8); // Copy

                uint32_t width = (uint32_t)pImage_bgr8->GetWidth();
                uint32_t height = (uint32_t)pImage_bgr8->GetHeight();
                uint32_t num_channels = 3;
                uint32_t data_size = width * height * num_channels * sizeof(uint8_t);

                // Sanity check
                if (data_size != rp->image_size) {
                    throw std::runtime_error("data_size != request_pool->image_size (expected: " + 
                        std::to_string(rp->image_size) + ", received: " + std::to_string(data_size) + ")");
                }

                memcpy(req_get_image_ptr(rp, req->idx), pImage_bgr8->GetData(), data_size);  // Copy, NOTE can we do a move? 
                req->image.width = width;
                req->image.height = height;
                req->image.num_channels = num_channels;
                req->image.data_offset = req_get_image_offset(rp, req->idx);
                req->image.data_size = data_size;

                Arena::ImageFactory::Destroy(pImage_bgr8); // destroy after processing
            }

            // 2. IMX501 Tensor processing
            // TODO: remove util and do it all ourselves

            // Extract the tensors data from the chunk data in a buffer
            // temporary assign a buffer of max size 
            size_t max_buffer_size = get_max_buffer_size(util);
            uint8_t *buffer = new uint8_t[max_buffer_size];
            extract_buffers(pChunkData, buffer, max_buffer_size);

            uint8_t *tensor_buffer = new uint8_t[max_buffer_size];
            extract_tensor_data(buffer, tensor_buffer, max_buffer_size, util);

            // Parse the buffer
            // Add the tensors to the request interface    
            parse_inference_data(tensor_buffer, max_buffer_size, req, rp, cfg->enable_input_tensor);

            // Cleanup
            delete[] buffer;
            buffer = nullptr;
            delete[] tensor_buffer;
            tensor_buffer = nullptr;

        }
    }
}


// -----------------------------------------------------------------------------
//  main
// -----------------------------------------------------------------------------

const std::map<unsigned int, std::array<unsigned int, 3>> IMAGE_SHAPE_MAP = {
    {1, std::array<unsigned int, 3>{SENSOR_W, SENSOR_H, 3}},           // RPS: 4+
    {2, std::array<unsigned int, 3>{2024, 1516, 3}},                   // RPS: 15+
    {3, std::array<unsigned int, 3>{1348, 1008, 3}},                   // RPS: 23+
    {4, std::array<unsigned int, 3>{1012, 756, 3}},                    // RPS: ~30
    {5, std::array<unsigned int, 3>{808, 604, 3}},                     // RPS: 30+
    {6, std::array<unsigned int, 3>{672, 504, 3}},                     // RPS: 30+
    {7, std::array<unsigned int, 3>{576, 432, 3}},                     // RPS: 30+
    {8, std::array<unsigned int, 3>{504, 378, 3}}                      // RPS: 30+
};


void start_triton(PyObject* py_callback) {
    try
    {
        // --------------------------INITIATE------------------------
        Arena::ISystem* pSystem = Arena::OpenSystem();
        pSystem->UpdateDevices(100);
        std::vector<Arena::DeviceInfo> deviceInfos = pSystem->GetDevices();
        if (deviceInfos.size() == 0)
            throw std::runtime_error("deviceInfos.size() == 0, no device connected");
        else {
            std::cout << "[INFO] Found number of devices: " << deviceInfos.size() << std::endl;
        }
        Arena::IDevice* pDevice = pSystem->CreateDevice(deviceInfos[0]);     
        std::cout  << "[INFO] Automatically selecting 1st device: "  << deviceInfos[0].ModelName() << ", " << deviceInfos[0].SerialNumber() << ", " << deviceInfos[0].IpAddressStr() << ".\n";
        
        // ...
        
        ArenaExample::IMX501Utils util(pDevice, false);

        // --------------------------SETTINGS------------------------
        TritonConfig* triton_config = get_shared_config();
        if (!triton_config) {
            std::cerr << "Failed to open shared triton config memory" << std::endl;
            return;
        }
        
        // Restore default camera settings
        util.SetCameraDefaults();

        // Init camera to output DNN & set DNN defaults
        bool suppress_raw_image_stream = triton_config->headless || triton_config->enable_input_tensor;
        util.InitCameraToOutputDNN(suppress_raw_image_stream);
        util.SetDNNDefaults();
        
        std::cout << "[INFO] suppress_raw_image_stream: " << suppress_raw_image_stream << std::endl;
        if (!suppress_raw_image_stream) {
            // Apply binning factor
            unsigned int binning_factor = static_cast<unsigned int>(triton_config->binning_factor);
            std::cout << "[INFO] binning factor: " << binning_factor << std::endl;
            util.SetBinning(binning_factor);
            
            // Image cropping
            uint32_t hires_l = triton_config->roi_hires[0];
            uint32_t hires_t = triton_config->roi_hires[1];
            uint32_t hires_w = triton_config->roi_hires[2];
            uint32_t hires_h = triton_config->roi_hires[3];

            const auto& shape = IMAGE_SHAPE_MAP.at(binning_factor);
            uint32_t MAX_WIDTH = shape[0];
            uint32_t MAX_HEIGHT = shape[1];

            // NOTE: important to first set the DNN image size, then the offset (otherwise constraints on max offset)
            if (hires_w != MAX_WIDTH || hires_h != MAX_HEIGHT ) { util.SetImageSize(hires_w, hires_h); }
            if (hires_l != 0 || hires_t != 0) { util.SetImageOffset(hires_l, hires_t); }
        }
        
        // Input tensor cropping
        uint32_t it_l = triton_config->roi_it[0];
        uint32_t it_t = triton_config->roi_it[1];
        uint32_t it_w = triton_config->roi_it[2];
        uint32_t it_h = triton_config->roi_it[3];

        // NOTE: important to first set the DNN image size, then the offset (otherwise constraints on max offset)
        if (it_w != SENSOR_W || it_h != SENSOR_H) { util.SetDNNImageSize(it_w, it_h); } // NOTE scaled to sensor resolution
        if (it_l != 0 || it_t != 0) { util.SetDNNImageOffset(it_l, it_t); }

        // Apply fps setting to camera (utility function is available)
        // Refer to the implementation for detailed procedure
        util.SetFPS(triton_config->frame_rate);

        // -------------------------GET SHARED REQUEST POOL------------------------
        RequestPool* request_pool = get_shared_request_pool(triton_config->total_pool_size);
        if (!request_pool) {
            std::cerr << "Failed to open shared request pool memory" << std::endl;
            return;
        }

        request_pool->init();       

        // --------------------------START STREAM------------------------
        pDevice->StartStream();

        while (triton_config->keep_running) {
            Arena::IImage *pImage;
            try
            {
                pImage = pDevice->GetImage(TIMEOUT);
            }
            catch (const GenICam::TimeoutException& e)
            {
                if (pDevice->IsConnected() != false)
                    continue;
                else
                {
                    std::cout << "The camera was disconnected" << std::endl;
                    break;
                }
            }

            // Process image & output tensors
            // Updates the RequestInterface & calls the python callback adding it to the request buffer
            if (pImage != nullptr)
            {
                if (pImage->IsIncomplete() == false)
                {
                    
                    // Get an available request from the pool
                    RequestInterface* req = request_pool->get_next_request();
                    if (req == nullptr) {
                        std::cout << "No available requests in pool, skipping frame" << std::endl;
                        pDevice->RequeueBuffer(pImage);
                        continue;
                    }
                    
                    // core processing
                    run(req, pImage, request_pool, util, triton_config);

                    // Ensure we have the GIL for Python callbacks
                    PyGILState_STATE gstate = PyGILState_Ensure();

                    // Call the Python callback
                    if (py_callback != nullptr) {
                        PyObject* py_args = PyTuple_New(1);
                        PyTuple_SetItem(py_args, 0, PyLong_FromLong((int)req->idx));
                        PyObject* result = PyObject_CallObject(py_callback, py_args);
                        if (result == nullptr) {
                            PyErr_Print();
                        } else {
                            // Py_DECREF(py_args);
                            Py_DECREF(result);
                        }
                    }

                    // Release the GIL
                    PyGILState_Release(gstate);

                    // request_pool->print_pool_usage();
                
                }
            }

            // Requeue buffer as soon as possible
            pDevice->RequeueBuffer(pImage);

            // Cleanup
            // requests are released from python when processed

        }
        pDevice->StopStream();

        request_pool->cleanup();

        // Restore initial settings
        util.SetCameraDefaults();
        util.SetDNNDefaults();

        pSystem->DestroyDevice(pDevice);
        Arena::CloseSystem(pSystem);
    
    } 
    catch (GenICam::GenericException &ge)
    {
        std::cout << "\nGenICam exception thrown: " << ge.what() << "\n";
    }
    catch (std::exception &ex)
    {
        std::cout << "\nStandard exception thrown: " << ex.what() << "\n";
    }
    catch (...)
    {
        std::cout << "\nUnexpected exception thrown\n";
    }
}


// -----------------------------------------------------------------------------
//  Get network info for denomarlizing the input tensor
// -----------------------------------------------------------------------------

inline int conv_reg_signed_12(uint16_t val) {
    return ((val >> 12) & 1) == 0 ? val : -((~val + 1) & 0x1FFF);
}

inline int conv_reg_signed_11(uint16_t val) {
    return ((val >> 11) & 1) == 0 ? val : -((~val + 1) & 0x0FFF);
}

DATNetworkInfo get_network_info(ArenaExample::IMX501Utils::fpk_info *fpk_info) {
    DATNetworkInfo info;
    info.is_dat_info = true;

    const auto& dnn = fpk_info->dnn[0];
    uint8_t input_format = dnn.input_tensor_format;

    // Input format: 0 = RGB, 1 = BGR
    if (input_format == 0 || input_format == 1) {
        // Map input_tensor_norm_k[] layout
        uint16_t K00 = dnn.input_tensor_norm_k[0];
        uint16_t K02 = dnn.input_tensor_norm_k[1];
        uint16_t K03 = dnn.input_tensor_norm_k[2];
        uint16_t K11 = dnn.input_tensor_norm_k[3];
        uint16_t K13 = dnn.input_tensor_norm_k[4];
        uint16_t K20 = dnn.input_tensor_norm_k[5];
        uint16_t K22 = dnn.input_tensor_norm_k[6];
        uint16_t K23 = dnn.input_tensor_norm_k[7];

        // norm_val from K03, K13, K23
        info.norm_val[0] = conv_reg_signed_12(K03);
        info.norm_val[1] = conv_reg_signed_12(K13);
        info.norm_val[2] = conv_reg_signed_12(K23);
        info.norm_shift[0] = 4;
        info.norm_shift[1] = 4;
        info.norm_shift[2] = 4;

        // div_val based on RGB or BGR
        if (input_format == 0) { // RGB
            info.div_val[0] = conv_reg_signed_11(K00);
            info.div_val[2] = conv_reg_signed_11(K22);
        } else { // BGR
            info.div_val[0] = conv_reg_signed_11(K02);
            info.div_val[2] = conv_reg_signed_11(K20);
        }
        info.div_val[1] = conv_reg_signed_11(K11);
        info.div_shift = 6;
    }

    return info;
}

// -----------------------------------------------------------------------------
//  Upload File
// -----------------------------------------------------------------------------

const char* get_file_name(ArenaExample::IMX501Utils::CameraFileType inFileType) {
    const char* fileName = "";
    
    switch (inFileType)
    {
        case ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_FIRMWARE:
            fileName = "DeepNeuralNetworkFirmware";
            break;
        case ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_LOADER:
            fileName = "DeepNeuralNetworkLoader";
            break;
        case ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_NETWORK:
            fileName = "DeepNeuralNetworkNetwork";
            break;
        case ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_INFO:
            fileName = "DeepNeuralNetworkInfo";
            break;
        case ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_CLASSIFICATION:
            fileName = "DeepNeuralNetworkClassification";
            break;
        default:
            throw std::runtime_error("inFileType == Unknown");
            break;
    }

    return fileName;
}


void write_label_file(ArenaExample::IMX501Utils& util, GenApi::INodeMap* pNodeMap, const std::vector<std::string>& labels) {
    // Calculate total size needed for the label file
    size_t total_size = 0;
    for (const auto& label : labels) {
        total_size += label.length() + 1; // +1 for newline
    }

    // Create buffer for all labels
    uint8_t* inFileBuf = new uint8_t[total_size];
    size_t current_pos = 0;

    // Copy each label to buffer with newlines
    for (const auto& label : labels) {
        memcpy(inFileBuf + current_pos, label.c_str(), label.length());
        current_pos += label.length();
        inFileBuf[current_pos++] = '\n'; // Add newline
    }

    // Get the camera file name for classification labels
    const char* label_camera_file_name = get_file_name(ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_CLASSIFICATION);

    // Write the label file to camera
    util.WriteFile(pNodeMap, label_camera_file_name, inFileBuf, total_size, 0, nullptr);

    // Cleanup
    delete[] inFileBuf;
}

std::vector<std::string> read_label_file(ArenaExample::IMX501Utils& util, GenApi::INodeMap* pNodeMap) {
    uint8_t *label_data;
    size_t label_data_size;
    std::vector<std::string> mLabelList;

    label_data = util.RetrieveLabelData(pNodeMap, &label_data_size);
    if (label_data_size != 0) {
        util.MakeLabelList(label_data, label_data_size, &mLabelList);
        delete[] label_data;
    }
    
    return mLabelList;
}


uint8_t *ReadLocalFile(const char *inFileName, size_t *outFileSize)
{
  struct stat st;
  if (stat(inFileName, &st) != 0)
  {
    std::cerr << "[ERROR] Can't access the file (stat): ";
    std::cerr << inFileName << "\n";
    return nullptr;
  }
  if ((st.st_mode & S_IFMT) != S_IFREG)
  {
    std::cerr << "[ERROR] (st.st_mode & S_IFMT) != S_IFREG\n";
    return nullptr;
  }

  *outFileSize = st.st_size;
  uint8_t* dataBuf = new uint8_t[*outFileSize];
  if (dataBuf == nullptr)
  {
    std::cerr << "[ERROR] dataBuf == nullptr\n";
    return nullptr;
  }

  FILE* fp;
#ifdef _WIN32
    if (fopen_s(&fp, inFileName, "rb") != 0) {
        fp = nullptr;
    }
#else
    fp = fopen(inFileName, "rb");
#endif

  if (fp == nullptr)
  {
    std::cerr << "[ERROR] Can't access the file (fopen): ";
    std::cerr << inFileName << "\n";
    delete[] dataBuf;
    return nullptr;
  }
  if (fread(dataBuf, 1, *outFileSize, fp) != *outFileSize)
  {
    std::cerr << "[ERROR] Can't access the file (fread): ";
    std::cerr << inFileName << "\n";
    delete[] dataBuf;
    return nullptr;
  }
  fclose(fp);

  return dataBuf;
}


std::string CalculateMD5(const uint8_t* data, size_t data_size)
{
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned int result_len;

    EVP_MD_CTX *mdctx = EVP_MD_CTX_create();
    const EVP_MD *md = EVP_md5();

    EVP_DigestInit_ex(mdctx, md, NULL);
    EVP_DigestUpdate(mdctx, data, data_size);
    EVP_DigestFinal_ex(mdctx, result, &result_len);

    EVP_MD_CTX_destroy(mdctx);

    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for(unsigned int i = 0; i < result_len; ++i)
        sout << std::setw(2) << static_cast<int>(result[i]);
    return sout.str();
}


void chunk_upload_network_noskip(
    ArenaExample::IMX501Utils& util,
    Arena::IDevice* pDevice,
    GenApi::INodeMap* pNodeMap,
    const char* camera_file_name,
    const uint8_t* dataBuf,
    size_t dataSize
) {

    // Upload the network file by comparing the md5 checksum of the deployed network and the provided file.
    // The md5 checksum of the currently deployed network is stored in the labels space on the camera.
    // Checksums are stored as a list of strings, where each string is the md5 checksum of a chunk of the network of lenght <buf_length>.
    // (Or the remainder of the network if the chunk size is smaller then <buf_length>)
    // <buf_length> represents the maximum allowed writable size of a chunk of data to the camera.
    std::vector<std::string> label_checksums = read_label_file(util, pNodeMap);

    util.SetEnumNode(pNodeMap, "FileSelector", camera_file_name);
    util.SetEnumNode(pNodeMap, "FileOperationSelector", "Open");
    util.SetEnumNode(pNodeMap, "FileOpenMode", "Write");
    util.ExecuteNode(pNodeMap, "FileOperationExecute");

    GenApi::CRegisterPtr pRegister = pNodeMap->GetNode("FileAccessBuffer");
    size_t buf_length = pRegister->GetLength();
    util.SetEnumNode(pNodeMap, "FileOperationSelector", "Write");

    // Checking md5 checksums
    // NOTE don't skip chunks for now, that is why we need a seperate loop to see if the network has changed
    // When skipping chunks is allowed one can combine this md5 checksum with the upload loop because you don't need the
    // final network_changed check from the beginning.
    size_t idx = 0;
    size_t offset = 0;
    size_t remainingSize = dataSize;
    const uint8_t* dataPtr = (uint8_t*)dataBuf;

    // bool update_chunk = false;
    bool network_changed = false;

    while (remainingSize != 0) {
        size_t upload_size = remainingSize;
        if (upload_size > buf_length)
            upload_size = buf_length;

        // MD5 CHECKSUM
        std::string md5_checksum = CalculateMD5(dataPtr + offset, upload_size);

        if (idx >= label_checksums.size()) {
            // adding new checksum label
            // update_chunk = true;
            network_changed = true;
            label_checksums.push_back(md5_checksum);
        } else {
            if (label_checksums[idx] == md5_checksum) {
                // equal checksums, no need to update
                // update_chunk = false;
            } else {
                // update the current label
                // update_chunk = true;
                network_changed = true;
                label_checksums[idx] = md5_checksum;
            }
        }

        // UPDATE
        remainingSize -= upload_size;
        offset += upload_size;
        idx++;
    }
    
    // Write the network file to the camera in chunks of <buf_length>
    offset = 0;
    remainingSize = dataSize;
    ProgressBar progress_bar(dataSize, 60);
    std::cout << "\n------------------------------------------------------------------------------------------------------------------\n"
            << "NOTE: Loading network firmware onto the IMX500 can take several minutes, please do not close down the application."
            << "\n------------------------------------------------------------------------------------------------------------------\n" << std::endl;
    progress_bar.update(0);


    while (remainingSize != 0)
    {
        size_t upload_size = remainingSize;
        if (upload_size > buf_length)
            upload_size = buf_length;

        // UPLOAD CHUNK IF NEEDED
        // NOTE: don't skip chunks for now, (possible bug)
        // if (update_chunk) {
        if (network_changed) { 
            util.SetIntNode(pNodeMap, "FileAccessOffset", offset);
            util.SetIntNode(pNodeMap, "FileAccessLength", (int64_t)upload_size);
            pRegister->Set(dataPtr + offset, upload_size);
            util.ExecuteNode(pNodeMap, "FileOperationExecute");
        }
        
        // UPDATE
        remainingSize -= upload_size;
        offset += upload_size;
        
        // PROGRESS FUNCTION    
        progress_bar.update(offset);
    }
    std::cout << "\n" << std::endl;

    util.SetEnumNode(pNodeMap, "FileOperationSelector", "Close");
    util.ExecuteNode(pNodeMap, "FileOperationExecute");

    //  if needed
    if (network_changed) {
        // Reload the network
        
        // BUG: reloading network here sometimes failes
        // no idea why, maybe allowing some extra time before it helps but doesn't solve the problem
        std::cout << "waiting some time ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        util.ReloadNetwork(pDevice); // Fails
        
        // Update labels with new checksums
        std::cout << "writing checksums ..." << std::endl;
        write_label_file(util, pNodeMap, label_checksums);
    }

    
}


void chunk_upload_network(
    ArenaExample::IMX501Utils& util,
    Arena::IDevice* pDevice,
    GenApi::INodeMap* pNodeMap,
    const char* camera_file_name,
    const uint8_t* dataBuf,
    size_t dataSize
) {

    // Upload the network file by comparing the md5 checksum of the deployed network and the provided file.
    // The md5 checksum of the currently deployed network is stored in the labels space on the camera.
    // Checksums are stored as a list of strings, where each string is the md5 checksum of a chunk of the network of lenght <buf_length>.
    // (Or the remainder of the network if the chunk size is smaller then <buf_length>)
    // <buf_length> represents the maximum allowed writable size of a chunk of data to the camera.
    std::vector<std::string> label_checksums = read_label_file(util, pNodeMap);

    util.SetEnumNode(pNodeMap, "FileSelector", camera_file_name);
    util.SetEnumNode(pNodeMap, "FileOperationSelector", "Open");
    util.SetEnumNode(pNodeMap, "FileOpenMode", "Write");
    util.ExecuteNode(pNodeMap, "FileOperationExecute");

    GenApi::CRegisterPtr pRegister = pNodeMap->GetNode("FileAccessBuffer");
    size_t buf_length = pRegister->GetLength();
    util.SetEnumNode(pNodeMap, "FileOperationSelector", "Write");

    // Checking md5 checksums
    // NOTE don't skip chunks for now, that is why we need a seperate loop to see if the network has changed
    // When skipping chunks is allowed one can combine this md5 checksum with the upload loop because you don't need the
    // final network_changed check from the beginning.
    size_t idx = 0;
    size_t offset = 0;
    size_t remainingSize = dataSize;
    const uint8_t* dataPtr = (uint8_t*)dataBuf;

    bool update_chunk = false;
    bool network_changed = false;

    while (remainingSize != 0) {
        size_t upload_size = remainingSize;
        if (upload_size > buf_length)
            upload_size = buf_length;

        // MD5 CHECKSUM
        std::string md5_checksum = CalculateMD5(dataPtr + offset, upload_size);

        if (idx >= label_checksums.size()) {
            // adding new checksum label
            update_chunk = true;
            network_changed = true;
            label_checksums.push_back(md5_checksum);
        } else {
            if (label_checksums[idx] == md5_checksum) {
                // equal checksums, no need to update
                update_chunk = false;
            } else {
                // update the current label
                update_chunk = true;
                network_changed = true;
                label_checksums[idx] = md5_checksum;
            }
        }

        // UPLOAD CHUNK IF NEEDED
        if (update_chunk) { 
            util.SetIntNode(pNodeMap, "FileAccessOffset", offset);
            util.SetIntNode(pNodeMap, "FileAccessLength", (int64_t)upload_size);
            pRegister->Set(dataPtr + offset, upload_size);
            util.ExecuteNode(pNodeMap, "FileOperationExecute");
        }

        // UPDATE
        remainingSize -= upload_size;
        offset += upload_size;
        idx++;

        // PROGRESS FUNCTION    
        std::cout << "Progress: " << (offset * 100 / dataSize) << "%" << std::endl;
    }

    util.SetEnumNode(pNodeMap, "FileOperationSelector", "Close");
    util.ExecuteNode(pNodeMap, "FileOperationExecute");

    //  if needed
    if (network_changed) {
        // Reload the network
        
        // BUG: reloading network here sometimes failes
        // no idea why, maybe allowing some extra time before it helps but doesn't solve the problem
        std::cout << "waiting some time ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        util.ReloadNetwork(pDevice); // Fails
        
        // Update labels with new checksums
        std::cout << "writing checksums ..." << std::endl;
        write_label_file(util, pNodeMap, label_checksums);
    }    
}


DATNetworkInfo triton_upload_file(const char* filePath, int fileType) {
    DATNetworkInfo info;
    info.is_dat_info = false;
    
    try
    {
        // --------------------------INITIATE------------------------
        Arena::ISystem* pSystem = Arena::OpenSystem();
        pSystem->UpdateDevices(100);
        std::vector<Arena::DeviceInfo> deviceInfos = pSystem->GetDevices();
        if (deviceInfos.size() == 0)
            throw std::runtime_error("deviceInfos.size() == 0, no device connected");
        else {
            // std::cout << "Found number of devices: " << deviceInfos.size() << std::endl;
        }
        
        Arena::IDevice* pDevice = pSystem->CreateDevice(deviceInfos[0]);     
        // std::cout  << "Automatically selecting 1st device: "  << deviceInfos[0].ModelName() << ", " << deviceInfos[0].SerialNumber() << ", " << deviceInfos[0].IpAddressStr() << ".\n";
        
        ArenaExample::IMX501Utils util(pDevice, false);
        // util.InitCameraToOutputDNN(true);

        GenApi::INodeMap *pNodeMap = pDevice->GetNodeMap();
        ArenaExample::IMX501Utils::CameraFileType camera_file_type = static_cast<ArenaExample::IMX501Utils::CameraFileType>(fileType);
        const char* camera_file_name = get_file_name(camera_file_type);
    
        // --------------------READ LOCAL FILE-----------------
        const uint8_t *dataBuf = nullptr;
        size_t dataSize = 0;

        dataBuf = ReadLocalFile(filePath, &dataSize);
        if (dataBuf == nullptr || dataSize == 0)
          return info;

        // --------------------UPLOAD STRATEGY-----------------
        if (camera_file_type == ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_FIRMWARE) {
            std::cout << "[INFO] Uploading network firmware ..." << std::endl;
            util.WriteFile(pNodeMap, camera_file_name, dataBuf, dataSize, 0, nullptr); // just upload file
        }
        else if (camera_file_type == ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_LOADER) {
            std::cout << "[INFO] Uploading network loader ..." << std::endl;
            util.WriteFile(pNodeMap, camera_file_name, dataBuf, dataSize, 0, nullptr); // just upload file
        }
        else if (camera_file_type == ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_INFO) {
            std::cout << "\033[33m[INFO] Retrieve network info (fpk_info.dat)\033[0m" << std::endl;
            util.WriteFile(pNodeMap, camera_file_name, dataBuf, dataSize, 0, nullptr); // just upload file

            // Reply with the content of the info file
            util.RetrieveFPKinfo(util.mNodeMap, &util.mFPKinfo);
            info = get_network_info(&util.mFPKinfo);
        }
        else if (camera_file_type == ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_CLASSIFICATION) {
            // Labels.txt space is reserved for md5 checksum of the deployed network.
            throw std::runtime_error("Uploading labels is not allowed. Labels.txt space is reserved for md5 checksum of the deployed network.");
        }
        else if (camera_file_type == ArenaExample::IMX501Utils::CameraFileType::FILE_DEEP_NEURAL_NETWORK_NETWORK) {
            

            // V0: just upload file
            // util.WriteFile(pNodeMap, camera_file_name, dataBuf, dataSize, 0, nullptr);
            // util.ReloadNetwork(pDevice); // Fails

            // V1: chunk upload network (no skipping chunks)
            // BUG: fails on reload network as well
            chunk_upload_network_noskip(util, pDevice, pNodeMap, camera_file_name, dataBuf, dataSize);

            // V2: chunk upload network
            // BUG: fails on reload network as well
            // chunk_upload_network(util, pDevice, pNodeMap, camera_file_name, dataBuf, dataSize);

        }
        else if (camera_file_type == ArenaExample::IMX501Utils::CameraFileType::CAMERA_FILE_TYPE_UNKNOWN) {
                throw std::runtime_error("inFileType == Unknown");
        }
        else {
                throw std::runtime_error("inFileType == Unknown");
        }

        // --------------------CLEANUP-----------------
        if (dataBuf != nullptr) {
            delete[] dataBuf;
            dataBuf = nullptr;
        }

        pSystem->DestroyDevice(pDevice);
        Arena::CloseSystem(pSystem);
    
    } 
    catch (GenICam::GenericException &ge)
    {
        std::cout << "\nGenICam exception thrown: \n";
        throw std::runtime_error(ge.what());
    }
    catch (std::exception &ex)
    {
        std::cout << "\nStandard exception thrown: \n";
        throw std::runtime_error(ex.what());
    }
    catch (...)
    {
        throw std::runtime_error("Unexpected exception thrown");
    }

    return info;
}


// -----------------------------------------------------------------------------
//  Simple Acquisition
// -----------------------------------------------------------------------------

void simple_acquisition() {
    try
    {
        // --------------------------INITIATE------------------------
        Arena::ISystem* pSystem = Arena::OpenSystem();
        pSystem->UpdateDevices(100);
        std::vector<Arena::DeviceInfo> deviceInfos = pSystem->GetDevices();
        if (deviceInfos.size() == 0)
            throw std::runtime_error("deviceInfos.size() == 0, no device connected");
        else {
            std::cout << "Found number of devices: " << deviceInfos.size() << std::endl;
        }

        std::cout << "Available devices:" << std::endl;
        for (size_t i = 0; i < deviceInfos.size(); i++) {
            std::cout << "Device " << i << ":" << std::endl;
            std::cout << "  VendorName: " << deviceInfos[i].VendorName() << std::endl;
            std::cout << "  Model: " << deviceInfos[i].ModelName() << std::endl;
            std::cout << "  Serial: " << deviceInfos[i].SerialNumber() << std::endl;
            std::cout << "  IP: " << deviceInfos[i].IpAddressStr() << std::endl;
            std::cout << "  SubnetMask: " << deviceInfos[i].SubnetMaskStr() << std::endl;
            std::cout << "  DefaultGateway: " << deviceInfos[i].DefaultGatewayStr() << std::endl;
            std::cout << "  MacAddress: " << deviceInfos[i].MacAddressStr() << std::endl;
        }

        Arena::IDevice* pDevice = pSystem->CreateDevice(deviceInfos[0]);     
        std::cout  << "Automatically selecting 1st device: "  << deviceInfos[0].ModelName() << ", " << deviceInfos[0].SerialNumber() << ", " << deviceInfos[0].IpAddressStr() << ".\n";
        
        // Acquire image
		pDevice->StartStream();

        Arena::IImage* pImage;
        try
        {
            pImage = pDevice->GetImage(TIMEOUT);
        }
        catch (const GenICam::TimeoutException& e)
        {
            std::cout << "\nGenICam timeout exception thrown: \n";
            std::cout << "Camera connected: " << pDevice->IsConnected() << std::endl;
            throw std::runtime_error(e.what());
        }

        std::cout << "\nImage acquired, size: (" << pImage->GetWidth() << ", " << pImage->GetHeight() << ")\n" << std::endl;
		
        // Clean up
        pDevice->RequeueBuffer(pImage);
		pDevice->StopStream();
		pSystem->DestroyDevice(pDevice);

        Arena::CloseSystem(pSystem);
        std::cout << "Simple acquisition completed" << std::endl;
    } 
    catch (GenICam::GenericException &ge)
    {
        std::cout << "\nGenICam exception thrown: \n";
        throw std::runtime_error(ge.what());
    }
    catch (std::exception &ex)
    {
        std::cout << "\nStandard exception thrown: \n";
        throw std::runtime_error(ex.what());
    }
    catch (...)
    {
        throw std::runtime_error("Unexpected exception thrown");
    }
}