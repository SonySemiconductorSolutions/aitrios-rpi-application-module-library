/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2024, Raspberry Pi Ltd
 *
 * imx500_tensor_parser.h - Parser for imx500 tensors
 */

// Modifications made by Sony Semiconductor Solutions Corp. 2025
// - Updated parser functions to work on the Triton Smart with IMX501 Intelligent Vision Sensor

#include "ArenaApi.h"
#include "IMX501Utils.h"


enum TensorType {
	InputTensor = 0,
	OutputTensor,
};


struct IMX500Tensors {
	bool valid;
	unsigned int offset;
};


struct Dimensions {
	uint8_t ordinal;
	uint16_t size;
	uint8_t serializationIndex;
	uint8_t padding;
};


struct IMX500OutputTensorInfo {
	uint32_t totalSize;
	uint32_t numTensors;
	std::string networkName;
	std::shared_ptr<float[]> data;
	std::vector<uint32_t> tensorDataNum;
	std::vector<std::vector<Dimensions>> vecDim;
	std::vector<uint32_t> numDimensions;
};


struct IMX500InputTensorInfo {
	unsigned int width;
	unsigned int height;
	unsigned int widthStride;
	unsigned int heightStride;
	unsigned int channels;
	unsigned int size;
	std::string networkName;
	std::shared_ptr<uint8_t[]> data;
};


size_t get_max_buffer_size(ArenaExample::IMX501Utils& util);
void extract_buffers(Arena::IChunkData *inChunkData, uint8_t *buffer, size_t buffer_size);
void extract_tensor_data(uint8_t *buffer, uint8_t *tensor_buffer, size_t buffer_size, ArenaExample::IMX501Utils& util);
void parse_inference_data(uint8_t *buffer, size_t buffer_size, RequestInterface *req, RequestPool* rp, bool enable_input_tensor);