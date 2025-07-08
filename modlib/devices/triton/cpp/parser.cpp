/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright (C) 2024, Raspberry Pi Ltd
 *
 * imx500_tensor_parser.cpp - Parser for imx500 tensors
 */

// Modifications made by Sony Semiconductor Solutions Corp. 2025
// - Updated parser functions to work on the Triton Smart with IMX501 Intelligent Vision Sensor

#include <cmath>
#include <future>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <array>

#include "ArenaApi.h"
#include "IMX501Utils.h"
#include "triton.h"
#include "parser.h"

#include "apParams.flatbuffers_generated.h"


// -----------------------------------------------------------------------------
//  TODO: move this 
// -----------------------------------------------------------------------------
size_t get_max_buffer_size(ArenaExample::IMX501Utils& util) {
    
    // Read fpk info 
    // TODO: this has been done by IMX501Utils::InitCameraToOutputDNN (so no need to do it again)
    // ArenaExample::IMX501Utils::fpk_info mFPKinfo;
    // util.RetrieveFPKinfo(util.mNodeMap, &mFPKinfo);
    // if (util.ValidateFPKinfo(&mFPKinfo) == false)
    //     throw std::runtime_error("Received invalid fpk_info");
    // TODO: try to get rid fo th reading the ch7 & ch8 values from a file !
    
    // Get max buffer size
    size_t mChunkWidth = (size_t)util.mFPKinfo.dnn[0].dd_ch7_x;
    size_t mChunkHeight = (size_t)util.mFPKinfo.dnn[0].dd_ch7_y + (size_t)util.mFPKinfo.dnn[0].dd_ch8_y;

    return mChunkWidth * mChunkHeight;
}


void extract_buffers(Arena::IChunkData *inChunkData, uint8_t *buffer, size_t buffer_size) {

    // Get the actual data size in the chunk of the part assigned to the DNN
    GenApi::CIntegerPtr pChunkDeepNeuralNetworkLength = inChunkData->GetChunk("ChunkDeepNeuralNetworkLength");
    if (!pChunkDeepNeuralNetworkLength ||
        !GenApi::IsAvailable(pChunkDeepNeuralNetworkLength) ||
        !GenApi::IsReadable(pChunkDeepNeuralNetworkLength))
    {
        throw GenICam::GenericException("missing chunk data [ChunkDeepNeuralNetworkLength]", __FILE__, __LINE__);
    }
    
    int64_t chunk_data_size = pChunkDeepNeuralNetworkLength->GetValue();
    if (chunk_data_size == 0)
        throw GenICam::GenericException("ChunkDeepNeuralNetworkLength is 0", __FILE__, __LINE__);
    if (chunk_data_size > (int64_t )buffer_size)
        throw GenICam::GenericException("ChunkDeepNeuralNetworkLength is bigger than the max buffer_size", __FILE__, __LINE__);

    // Fill the buffer with the DNN Chunk data
    GenApi::CRegisterPtr pChunkDeepNeuralNetwork = inChunkData->GetChunk("ChunkDeepNeuralNetwork");
    if (!pChunkDeepNeuralNetwork ||
        !GenApi::IsAvailable(pChunkDeepNeuralNetwork) ||
        !GenApi::IsReadable(pChunkDeepNeuralNetwork))
    {
        throw GenICam::GenericException("missing chunk data [ChunkDeepNeuralNetwork]", __FILE__, __LINE__);
    }

    pChunkDeepNeuralNetwork->Get(buffer, chunk_data_size);
}


constexpr unsigned int TensorStride = 2560;


void extract_tensor_data(uint8_t *buffer, uint8_t *tensor_buffer, size_t buffer_size, ArenaExample::IMX501Utils& util) {

	size_t lineLen = (size_t)util.mFPKinfo.dnn[0].dd_ch7_x;
	size_t lineNum = (size_t)util.mFPKinfo.dnn[0].dd_ch7_y + (size_t)util.mFPKinfo.dnn[0].dd_ch8_y;
	
	// ASSUMPTION this is equal to the TensorStride
	size_t dataWidth = TensorStride; // == inHeader->max_length_of_line;
	// TODO check if that is true

	// sanity check
	if (buffer == nullptr || tensor_buffer == nullptr ||
		lineLen * lineNum > buffer_size ||
		dataWidth > lineLen) {
		std::cout << "[ERROR] ExtractTensorData was called with wrong parameters" << std::endl;
	}

	for (size_t i = 0; i < lineNum; i++)
	{
		memcpy(tensor_buffer, buffer, dataWidth);
		buffer += lineLen;
		tensor_buffer += dataWidth;
	}

}


// -----------------------------------------------------------------------------
//  Parser
// -----------------------------------------------------------------------------



constexpr unsigned int DnnHeaderSize = 12;
constexpr unsigned int MipiPhSize = 0;
constexpr unsigned int InputSensorMaxWidth = 1280;
constexpr unsigned int InputSensorMaxHeight = 960;


enum TensorDataType {
	Signed = 0,
	Unsigned
};


struct DnnHeader {
	uint8_t frameValid;
	uint8_t frameCount;
	uint16_t maxLineLen;
	uint16_t apParamSize;
	uint16_t networkId;
	uint8_t tensorType;
    uint8_t reserved[3];
};


struct OutputTensorApParams {
	uint8_t id;
	std::string name;
	std::string networkName;
	uint16_t numDimensions;
	uint8_t bitsPerElement;
	std::vector<Dimensions> vecDim;
	uint16_t shift;
	float scale;
	uint8_t format;
};


struct InputTensorApParams {
	uint8_t networkId;
	std::string networkName;
	uint16_t width;
	uint16_t height;
	uint16_t channel;
	uint16_t widthStride;
	uint16_t heightStride;
	uint8_t format;
};


// -----------------------------------------------------------------------------
//  Core parsing functions
// -----------------------------------------------------------------------------

int parseHeader(DnnHeader &dnnHeader, std::vector<uint8_t> &apParams, const uint8_t *src)
{
	dnnHeader = *reinterpret_cast<const DnnHeader *>(src);

	if (!dnnHeader.frameValid)
		return -1;

	apParams.resize(dnnHeader.apParamSize, 0);

	uint32_t i = DnnHeaderSize;
	for (unsigned int j = 0; j < dnnHeader.apParamSize; j++) {
		if (i >= TensorStride) {
			i = 0;
			src += TensorStride + MipiPhSize;
		}
		apParams[j] = src[i++];
	}

	return 0;
}


int parseOutputApParams(std::vector<OutputTensorApParams> &outputApParams, const std::vector<uint8_t> &apParams,
			const DnnHeader &dnnHeader)
{
	const apParams::fb::FBApParams *fbApParams;
	const apParams::fb::FBNetwork *fbNetwork;
	const apParams::fb::FBOutputTensor *fbOutputTensor;

	fbApParams = apParams::fb::GetFBApParams(apParams.data());
	// std::cout << "[DEBUG] Networks size: " << fbApParams->networks()->size() << std::endl;

	outputApParams.clear();

	for (unsigned int i = 0; i < fbApParams->networks()->size(); i++) {
		fbNetwork = (apParams::fb::FBNetwork *)(fbApParams->networks()->Get(i));
		if (fbNetwork->id() != dnnHeader.networkId)
			continue;

		// std::cout << "[DEBUG] "
		// 	<< "Network: " << fbNetwork->type()->c_str()
		// 	<< ", i/p size: " << fbNetwork->inputTensors()->size()
		// 	<< ", o/p size: " << fbNetwork->outputTensors()->size() << std::endl;

		for (unsigned int j = 0; j < fbNetwork->outputTensors()->size(); j++) {
			OutputTensorApParams outApParam;

			fbOutputTensor = (apParams::fb::FBOutputTensor *)fbNetwork->outputTensors()->Get(j);

			outApParam.id = fbOutputTensor->id();
			outApParam.name = fbOutputTensor->name()->str();
			outApParam.networkName = fbNetwork->type()->str();
			outApParam.numDimensions = fbOutputTensor->numOfDimensions();

			for (unsigned int k = 0; k < fbOutputTensor->numOfDimensions(); k++) {
				Dimensions dim;
				dim.ordinal = fbOutputTensor->dimensions()->Get(k)->id();
				dim.size = fbOutputTensor->dimensions()->Get(k)->size();
				dim.serializationIndex = fbOutputTensor->dimensions()->Get(k)->serializationIndex();
				dim.padding = fbOutputTensor->dimensions()->Get(k)->padding();
				if (dim.padding != 0) {
					std::cout << "[ERROR] " 
						<< "Error in AP Params, Non-Zero padding for Dimension " << k << std::endl;
					return -1;
				}

				outApParam.vecDim.push_back(dim);
			}

			outApParam.bitsPerElement = fbOutputTensor->bitsPerElement();
			outApParam.shift = fbOutputTensor->shift();
			outApParam.scale = fbOutputTensor->scale();
			outApParam.format = fbOutputTensor->format();

			/* Add the element to vector */
			outputApParams.push_back(outApParam);
		}

		break;
	}

	return 0;
}


int populateOutputTensorInfo(IMX500OutputTensorInfo &outputTensorInfo,
			     const std::vector<OutputTensorApParams> &outputApParams)
{
	/* Calculate total output size. */
	unsigned int totalOutSize = 0;
	for (auto const &ap : outputApParams) {
		unsigned int totalDimensionSize = 1;
		for (auto &dim : ap.vecDim) {
			if (totalDimensionSize >= std::numeric_limits<uint32_t>::max() / dim.size) {
				std::cout << "[ERROR] Invalid totalDimensionSize" << std::endl;
				return -1;
			}

			totalDimensionSize *= dim.size;
		}

		if (totalOutSize >= std::numeric_limits<uint32_t>::max() - totalDimensionSize) {
			std::cout << "[ERROR] Invalid totalOutSize" << std::endl;
			return -1;
		}

		totalOutSize += totalDimensionSize;
	}

	if (totalOutSize == 0) {
		std::cout << "[ERROR] Invalid output tensor info (totalOutSize is 0)" << std::endl;
		return -1;
	}

	// std::cout << "[DEBUG] Final output size: " << totalOutSize << std::endl;

	if (totalOutSize >= std::numeric_limits<uint32_t>::max() / sizeof(float)) {
		std::cout << "[ERROR] Invalid output tensor info" << std::endl;
		return -1;
	}

	outputTensorInfo.data = std::shared_ptr<float[]>(new float[totalOutSize]);
	unsigned int numOutputTensors = static_cast<unsigned int>(outputApParams.size());

	if (!numOutputTensors) {
		std::cout << "[ERROR] Invalid numOutputTensors (0)" << std::endl;
		return -1;
	}

	if (numOutputTensors >= std::numeric_limits<uint32_t>::max() / sizeof(uint32_t)) {
		std::cout << "[ERROR] Invalid numOutputTensors" << std::endl;
		return -1;
	}

	outputTensorInfo.totalSize = totalOutSize;
	outputTensorInfo.numTensors = numOutputTensors;
	outputTensorInfo.networkName = outputApParams[0].networkName;
	outputTensorInfo.tensorDataNum.resize(numOutputTensors, 0);
	for (auto const &p : outputApParams) {
		outputTensorInfo.vecDim.push_back(p.vecDim);
		outputTensorInfo.numDimensions.push_back(static_cast<uint32_t>(p.vecDim.size()));
	}

	return 0;
}



template<typename T>
float getVal8(const uint8_t *src, const OutputTensorApParams &param)
{
	T temp = (T)*src;
	float value = (temp - param.shift) * param.scale;
	return value;
}

template<typename T>
float getVal16(const uint8_t *src, const OutputTensorApParams &param)
{
	T temp = (((T) * (src + 1)) & 0xff) << 8 | (*src & 0xff);
	float value = (temp - param.shift) * param.scale;
	return value;
}

template<typename T>
float getVal32(const uint8_t *src, const OutputTensorApParams &param)
{
	T temp = (((T) * (src + 3)) & 0xff) << 24 | (((T) * (src + 2)) & 0xff) << 16 |
		 (((T) * (src + 1)) & 0xff) << 8 | (*src & 0xff);
	float value = (temp - param.shift) * param.scale;
	return value;
}


int parseOutputTensorBody(IMX500OutputTensorInfo &outputTensorInfo, const uint8_t *src,
			  const std::vector<OutputTensorApParams> &outputApParams,
			  const DnnHeader &dnnHeader)
{
	float *dst = outputTensorInfo.data.get();
	int ret = 0;

	if (outputTensorInfo.totalSize > (std::numeric_limits<uint32_t>::max() / sizeof(float))) {
		std::cout << "[ERROR] totalSize is greater than maximum size" << std::endl;
		return -1;
	}

	std::unique_ptr<float[]> tmpDst = std::make_unique<float[]>(outputTensorInfo.totalSize);
	std::vector<uint16_t> numLinesVec(outputApParams.size());
	std::vector<uint32_t> outSizes(outputApParams.size());
	std::vector<uint32_t> offsets(outputApParams.size());
	std::vector<const uint8_t *> srcArr(outputApParams.size());
	std::vector<std::vector<Dimensions>> serializedDims;
	std::vector<std::vector<Dimensions>> actualDims;

	const uint8_t *src1 = src;
	uint32_t offset = 0;
	std::vector<Dimensions> serializedDimT;
	std::vector<Dimensions> actualDimT;

	for (unsigned int tensorIdx = 0; tensorIdx < outputApParams.size(); tensorIdx++) {
		offsets[tensorIdx] = offset;
		srcArr[tensorIdx] = src1;
		uint32_t tensorDataNum = 0;

		const OutputTensorApParams &param = outputApParams.at(tensorIdx);
		uint32_t outputTensorSize = 0;
		uint32_t tensorOutSize = (param.bitsPerElement / 8);

		serializedDimT.resize(param.numDimensions);
		actualDimT.resize(param.numDimensions);

		for (int idx = 0; idx < param.numDimensions; idx++) {
			actualDimT[idx].size = param.vecDim.at(idx).size;
			serializedDimT[param.vecDim.at(idx).serializationIndex].size = param.vecDim.at(idx).size;

			tensorOutSize *= param.vecDim.at(idx).size;
			if (tensorOutSize >= std::numeric_limits<uint32_t>::max() / param.bitsPerElement / 8) {
				std::cout << "[ERROR] Invalid output tensor info" << std::endl;
				return -1;
			}

			actualDimT[idx].serializationIndex = param.vecDim.at(idx).serializationIndex;
			serializedDimT[param.vecDim.at(idx).serializationIndex].serializationIndex =
				static_cast<uint8_t>(idx);
		}

		uint16_t numLines = static_cast<uint16_t>(std::ceil(tensorOutSize / static_cast<float>(dnnHeader.maxLineLen)));
		outputTensorSize = tensorOutSize;
		numLinesVec[tensorIdx] = numLines;
		outSizes[tensorIdx] = tensorOutSize;

		serializedDims.push_back(serializedDimT);
		actualDims.push_back(actualDimT);

		src1 += numLines * TensorStride;
		tensorDataNum = (outputTensorSize / (param.bitsPerElement / 8));
		offset += tensorDataNum;
		outputTensorInfo.tensorDataNum[tensorIdx] = tensorDataNum;
		if (offset > outputTensorInfo.totalSize) {
			std::cout << "[ERROR] "
				<< "Error in parsing output tensor offset " << offset << " > output_size" << std::endl;
			return -1;
		}
	}

	std::vector<uint32_t> idxs(outputApParams.size());
	for (unsigned int i = 0; i < idxs.size(); i++)
		idxs[i] = i;

	for (unsigned int i = 0; i < idxs.size(); i++) {
		for (unsigned int j = 0; j < idxs.size(); j++) {
			if (numLinesVec[idxs[i]] > numLinesVec[idxs[j]])
				std::swap(idxs[i], idxs[j]);
		}
	}

	std::vector<std::future<int>> futures;
	for (unsigned int ii = 0; ii < idxs.size(); ii++) {
		uint32_t idx = idxs[ii];
		futures.emplace_back(std::async(
			std::launch::async,
			[&tmpDst, &outSizes, &numLinesVec, &actualDims, &serializedDims,
			 &outputApParams, &dnnHeader, dst](int tensorIdx, const uint8_t *tsrc, int toffset) -> int {
				uint32_t outputTensorSize = outSizes[tensorIdx];
				uint16_t numLines = numLinesVec[tensorIdx];
				bool sortingRequired = false;

				const OutputTensorApParams &param = outputApParams[tensorIdx];
				const std::vector<Dimensions> &serializedDim = serializedDims[tensorIdx];
				const std::vector<Dimensions> &actualDim = actualDims[tensorIdx];

				for (unsigned i = 0; i < param.numDimensions; i++) {
					if (param.vecDim.at(i).serializationIndex != param.vecDim.at(i).ordinal)
						sortingRequired = true;
				}

				if (!outputTensorSize) {
					std::cout << "[ERROR] Invalid output tensorsize (0)" << std::endl;
					return -1;
				}

				/* Extract output tensor data */
				uint32_t elementIndex = 0;
				if (param.bitsPerElement == 8) {
					for (unsigned int i = 0; i < numLines; i++) {
						int lineIndex = 0;
						while (lineIndex < dnnHeader.maxLineLen) {
							if (param.format == TensorDataType::Signed)
								tmpDst[toffset + elementIndex] =
									getVal8<int8_t>(tsrc + lineIndex, param);
							else
								tmpDst[toffset + elementIndex] =
									getVal8<uint8_t>(tsrc + lineIndex, param);
							elementIndex++;
							lineIndex++;
							if (elementIndex == outputTensorSize)
								break;
						}
						tsrc += TensorStride;
						if (elementIndex == outputTensorSize)
							break;
					}
				} else if (param.bitsPerElement == 16) {
					for (unsigned int i = 0; i < numLines; i++) {
						int lineIndex = 0;
						while (lineIndex < dnnHeader.maxLineLen) {
							if (param.format == TensorDataType::Signed)
								tmpDst[toffset + elementIndex] =
									getVal16<int16_t>(tsrc + lineIndex, param);
							else
								tmpDst[toffset + elementIndex] =
									getVal16<uint16_t>(tsrc + lineIndex, param);
							elementIndex++;
							lineIndex += 2;
							if (elementIndex >= (outputTensorSize >> 1))
								break;
						}
						tsrc += TensorStride;
						if (elementIndex >= (outputTensorSize >> 1))
							break;
					}
				} else if (param.bitsPerElement == 32) {
					for (unsigned int i = 0; i < numLines; i++) {
						int lineIndex = 0;
						while (lineIndex < dnnHeader.maxLineLen) {
							if (param.format == TensorDataType::Signed)
								tmpDst[toffset + elementIndex] =
									getVal32<int32_t>(tsrc + lineIndex, param);
							else
								tmpDst[toffset + elementIndex] =
									getVal32<uint32_t>(tsrc + lineIndex, param);
							elementIndex++;
							lineIndex += 4;
							if (elementIndex >= (outputTensorSize >> 2))
								break;
						}
						tsrc += TensorStride;
						if (elementIndex >= (outputTensorSize >> 2))
							break;
					}
				}

				/*
				 * Sorting in order according to AP Params. Not supported if larger than 3D
				 * Preparation:
				 */
				if (sortingRequired) {
					constexpr unsigned int DimensionMax = 3;

					std::array<uint32_t, DimensionMax> loopCnt{ 1, 1, 1 };
					std::array<uint32_t, DimensionMax> coef{ 1, 1, 1 };
					for (unsigned int i = 0; i < param.numDimensions; i++) {
						if (i >= DimensionMax) {
							std::cout << "[ERROR] numDimensions value is 3 or higher" << std::endl;
							break;
						}

						loopCnt[i] = serializedDim.at(i).size;

						for (unsigned int j = serializedDim.at(i).serializationIndex; j > 0; j--)
							coef[i] *= actualDim.at(j - 1).size;
					}
					/* Sort execution */
					unsigned int srcIndex = 0;
					unsigned int dstIndex;
					for (unsigned int i = 0; i < loopCnt[DimensionMax - 1]; i++) {
						for (unsigned int j = 0; j < loopCnt[DimensionMax - 2]; j++) {
							for (unsigned int k = 0; k < loopCnt[DimensionMax - 3]; k++) {
								dstIndex = (coef[DimensionMax - 1] * i) +
									   (coef[DimensionMax - 2] * j) +
									   (coef[DimensionMax - 3] * k);
								dst[toffset + dstIndex] = tmpDst[toffset + srcIndex++];
							}
						}
					}
				} else {
					if (param.bitsPerElement == 8)
						memcpy(dst + toffset, tmpDst.get() + toffset,
						       outputTensorSize * sizeof(float));
					else if (param.bitsPerElement == 16)
						memcpy(dst + toffset, tmpDst.get() + toffset,
						       (outputTensorSize >> 1) * sizeof(float));
					else if (param.bitsPerElement == 32)
						memcpy(dst + toffset, tmpDst.get() + toffset,
						       (outputTensorSize >> 2) * sizeof(float));
					else {
						std::cout << "[ERROR] "
							<< "Invalid bitsPerElement value =" << param.bitsPerElement << std::endl;
						return -1;
					}
				}

				return 0;
			},
			idx, srcArr[idx], offsets[idx]));
	}

	for (auto &f : futures)
		ret += f.get();

	return ret;
}


int parseInputApParams(InputTensorApParams &inputApParams, const std::vector<uint8_t> &apParams,
		       const DnnHeader &dnnHeader)
{
	const apParams::fb::FBApParams *fbApParams;
	const apParams::fb::FBNetwork *fbNetwork;
	const apParams::fb::FBInputTensor *fbInputTensor;

	fbApParams = apParams::fb::GetFBApParams(apParams.data());
	// std::cout << "[DEBUG] Networks size: " << fbApParams->networks()->size() << std::endl;

	for (unsigned int i = 0; i < fbApParams->networks()->size(); i++) {
		fbNetwork = reinterpret_cast<const apParams::fb::FBNetwork *>(fbApParams->networks()->Get(i));
		if (fbNetwork->id() != dnnHeader.networkId)
			continue;

		// std::cout << "[DEBUG] "
		// 	<< "Network: " << fbNetwork->type()->c_str()
		// 	<< ", i/p size: " << fbNetwork->inputTensors()->size()
		// 	<< ", o/p size: " << fbNetwork->outputTensors()->size() << std::endl;

		inputApParams.networkName = fbNetwork->type()->str();
		fbInputTensor =
			reinterpret_cast<const apParams::fb::FBInputTensor *>(fbNetwork->inputTensors()->Get(0));

		// std::cout << "[DEBUG] "
		// 	<< "Input Tensor shift: " << fbInputTensor->shift()
		// 	<< ", Scale: scale: " << fbInputTensor->scale()
		// 	<< ", Format: " << static_cast<int>(fbInputTensor->format()) << std::endl;

		if (fbInputTensor->dimensions()->size() != 3) {
			std::cout << "[ERROR] Invalid number of dimensions in InputTensor" << std::endl;
			return -1;
		}

		for (unsigned int j = 0; j < fbInputTensor->dimensions()->size(); j++) {
			switch (fbInputTensor->dimensions()->Get(j)->serializationIndex()) {
			case 0:
				inputApParams.width = fbInputTensor->dimensions()->Get(j)->size();
				inputApParams.widthStride =
					inputApParams.width + fbInputTensor->dimensions()->Get(j)->padding();
				break;
			case 1:
				inputApParams.height = fbInputTensor->dimensions()->Get(j)->size();
				inputApParams.heightStride =
					inputApParams.height + fbInputTensor->dimensions()->Get(j)->padding();
				break;
			case 2:
				inputApParams.channel = fbInputTensor->dimensions()->Get(j)->size();
				break;
			default:
				std::cout << "[ERROR] Invalid dimension in InputTensor " << j << std::endl;
				break;
			}
		}
	}

	return 0;
}

int parseInputTensorBody(IMX500InputTensorInfo &inputTensorInfo, const uint8_t *src,
			 const InputTensorApParams &inputApParams, const DnnHeader &dnnHeader)
{
	if ((inputApParams.width > InputSensorMaxWidth) || (inputApParams.height > InputSensorMaxHeight) ||
	    ((inputApParams.channel != 1) && (inputApParams.channel != 3) && (inputApParams.channel != 4))) {
		std::cout << "[ERROR] "
			<< "Invalid input tensor size w: " << inputApParams.width
			<< " h: " << inputApParams.height
			<< " c: " << inputApParams.channel << std::endl;
		return -1;
	}

	unsigned int outSize = inputApParams.width * inputApParams.height * inputApParams.channel;
	unsigned int outSizePadded = inputApParams.widthStride * inputApParams.heightStride * inputApParams.channel;
	unsigned int numLines = std::ceil(outSizePadded / static_cast<float>(dnnHeader.maxLineLen));
	inputTensorInfo.data = std::shared_ptr<uint8_t[]>(new uint8_t[outSize]);

	unsigned int diff = 0, outLineIndex = 0, pixelIndex = 0, heightIndex = 0, size = 0, left = 0;
	unsigned int wPad = inputApParams.widthStride - inputApParams.width;
	unsigned int hPad = inputApParams.heightStride - inputApParams.height;

	for (unsigned int line = 0; line < numLines; line++) {
		for (unsigned int lineIndex = diff; lineIndex < dnnHeader.maxLineLen; lineIndex += size) {
			if (outLineIndex == inputApParams.width) { /* Skip width padding pixels */
				outLineIndex = 0;
				heightIndex++;
				lineIndex += wPad;
				if (lineIndex >= dnnHeader.maxLineLen) {
					diff = lineIndex - dnnHeader.maxLineLen;
					break;
				} else
					diff = 0;
			}

			if (heightIndex == inputApParams.height) { /* Skip height padding pixels */
				lineIndex += hPad * inputApParams.widthStride;
				heightIndex = 0;
				if (lineIndex >= dnnHeader.maxLineLen) {
					diff = lineIndex - dnnHeader.maxLineLen;
					while (diff >= dnnHeader.maxLineLen) {
						diff -= dnnHeader.maxLineLen;
						src += TensorStride;
						line++;
					}
					break;
				} else
					diff = 0;
			}

			if (((pixelIndex == inputApParams.width * inputApParams.height) ||
			     (pixelIndex == inputApParams.width * inputApParams.height * 2) ||
			     (pixelIndex == inputApParams.width * inputApParams.height * 3))) {
				if (pixelIndex == outSize)
					break;
			}

			if (left > 0) {
				size = left;
				left = 0;
			} else if (pixelIndex + inputApParams.width >= outSize) {
				size = outSize - pixelIndex;
			} else if (lineIndex + inputApParams.width >= dnnHeader.maxLineLen) {
				size = dnnHeader.maxLineLen - lineIndex;
				left = inputApParams.width - size;
			} else {
				size = inputApParams.width;
			}

			memcpy(&inputTensorInfo.data[pixelIndex], src + lineIndex, size);
			pixelIndex += size;
			outLineIndex += size;
		}

		if (pixelIndex == outSize)
			break;

		src += TensorStride;
	}

	inputTensorInfo.size = outSize;
	inputTensorInfo.width = inputApParams.width;
	inputTensorInfo.height = inputApParams.height;
	inputTensorInfo.channels = inputApParams.channel;
	inputTensorInfo.widthStride = inputApParams.widthStride;
	inputTensorInfo.heightStride = inputApParams.heightStride;
	inputTensorInfo.networkName = inputApParams.networkName;

	return 0;
}


// -----------------------------------------------------------------------------
//  Utilities
// -----------------------------------------------------------------------------


void print_dnn_header(const DnnHeader &dnnHeader) {
	std::cout << "[DEBUG] Header: valid " << static_cast<bool>(dnnHeader.frameValid)
		<< " count " << static_cast<int>(dnnHeader.frameCount)
		<< " max len " << dnnHeader.maxLineLen
		<< " ap param size " << dnnHeader.apParamSize
		<< " network id " << dnnHeader.networkId
		<< " tensor type " << static_cast<int>(dnnHeader.tensorType) << std::endl;
}

void print_input_ap_params(const InputTensorApParams &inputApParams) {

	std::cout << "[INPUT AP PARAMS] "
		<< "Network ID: " << static_cast<int>(inputApParams.networkId)
		<< ", Network Name: " << inputApParams.networkName
		<< ", Width: " << static_cast<int>(inputApParams.width)
		<< ", Height: " << static_cast<int>(inputApParams.height)
		<< ", Channel: " << static_cast<int>(inputApParams.channel)
		<< ", Width Stride: " << static_cast<int>(inputApParams.widthStride)
		<< ", Height Stride: " << static_cast<int>(inputApParams.heightStride)
		<< ", Format: " << static_cast<int>(inputApParams.format)
		<< std::endl;

};


void print_output_ap_params(const std::vector<OutputTensorApParams> &outputApParams) {
		
	for (const auto &param : outputApParams) {
		std::cout << "[OUTPUT AP PARAMS] "
			<< "ID: " << static_cast<int>(param.id)
			<< ", Name: " << param.name
			<< ", Network Name: " << param.networkName
			<< ", Num Dimensions: " << static_cast<int>(param.numDimensions)
			<< ", Bits Per Element: " << static_cast<int>(param.bitsPerElement)
			<< std::endl;
		
		for (const auto &dim : param.vecDim) {
			std::cout << "    - Ordinal: " << static_cast<int>(dim.ordinal)
				<< ", Size: " << static_cast<int>(dim.size)
				<< ", Serialization Index: " << static_cast<int>(dim.serializationIndex)
				<< ", Padding: " << static_cast<int>(dim.padding)
				<< std::endl;
		}

		std::cout << "    Shift: " << static_cast<int>(param.shift)
			<< ", Scale: " << param.scale
			<< ", Format: " << static_cast<int>(param.format)
			<< std::endl;
	}
}


void print_output_tensor_info(const IMX500OutputTensorInfo &outputTensorInfo) {

	std::cout << "[OUTPUT TENSOR INFO] "
		<< "Total Size: " << static_cast<int>(outputTensorInfo.totalSize)
		<< ", Num Tensors: " << static_cast<int>(outputTensorInfo.numTensors)
		<< ", Network Name: " << outputTensorInfo.networkName
		<< std::endl;

	for (unsigned int i = 0; i < outputTensorInfo.numTensors; i++) {
		std::cout << "    - Tensor " << i << ": "
			<< "Data Num: " << static_cast<int>(outputTensorInfo.tensorDataNum[i])
			<< ", Num Dimensions: " << static_cast<int>(outputTensorInfo.numDimensions[i])
			<< std::endl;

		for (unsigned int j = 0; j < outputTensorInfo.numDimensions[i]; j++) {
			std::cout << "        - Dimension " << j << ": "
				<< "Ordinal: " << static_cast<int>(outputTensorInfo.vecDim[i][j].ordinal)
				<< ", Size: " << static_cast<int>(outputTensorInfo.vecDim[i][j].size)
				<< ", Serialization Index: " << static_cast<int>(outputTensorInfo.vecDim[i][j].serializationIndex)
				<< ", Padding: " << static_cast<int>(outputTensorInfo.vecDim[i][j].padding)
				<< std::endl;
		}
	}
		
		

}

// -----------------------------------------------------------------------------
//  ...
// -----------------------------------------------------------------------------


int imx500ParseInputTensor(IMX500InputTensorInfo &inputTensorInfo, uint8_t *inputTensor) {
	
    DnnHeader dnnHeader;
	std::vector<uint8_t> apParams;
	InputTensorApParams inputApParams{};

	int ret = parseHeader(dnnHeader, apParams, inputTensor);
	if (ret) {
		std::cout << "[ERROR] Header param parsing failed!" << std::endl;
		return ret;
	}

	if (dnnHeader.tensorType != TensorType::InputTensor) {
		std::cout << "[ERROR] Invalid input tensor type in AP params!" << std::endl;
		return -1;
	}

	ret = parseInputApParams(inputApParams, apParams, dnnHeader);
	if (ret) {
		std::cout << "[ERROR] AP param parsing failed!" << std::endl;
		return ret;
	}

	// print_input_ap_params(inputApParams);

	ret = parseInputTensorBody(inputTensorInfo, inputTensor + TensorStride, inputApParams, dnnHeader);
	if (ret) {
		std::cout << "[ERROR] Input tensor body parsing failed!" << std::endl;
		return ret;
	}

	return 0;
}


int imx500ParseOutputTensor(IMX500OutputTensorInfo &outputTensorInfo, uint8_t *outputTensor) {

    DnnHeader dnnHeader;
    std::vector<uint8_t> apParams;
    std::vector<OutputTensorApParams> outputApParams;

	int ret = parseHeader(dnnHeader, apParams, outputTensor);
	if (ret) {
		std::cout << "[ERROR] Header param parsing failed!" << std::endl;
		return ret;
	}

	if (dnnHeader.tensorType != TensorType::OutputTensor) {
		std::cout << "[ERROR] Invalid output tensor type in AP params!" << std::endl;
		return -1;
	}

	ret = parseOutputApParams(outputApParams, apParams, dnnHeader);
	if (ret) {
		std::cout << "[ERROR] AP param parsing failed!" << std::endl;
		return ret;
	}

	// print_output_ap_params(outputApParams);

	ret = populateOutputTensorInfo(outputTensorInfo, outputApParams);
	if (ret) {
		std::cout << "[ERROR] Failed to populate OutputTensorInfo!" << std::endl;
		return ret;
	}

	// print_output_tensor_info(outputTensorInfo);

	ret = parseOutputTensorBody(outputTensorInfo, outputTensor + TensorStride, outputApParams, dnnHeader);
	if (ret) {
		std::cout << "[ERROR] Output tensor body parsing failed!" << std::endl;
		return ret;
	}

    return 0;
}


std::unordered_map<TensorType, IMX500Tensors> get_tensor_offsets(uint8_t *buffer, size_t buffer_size) {

    const DnnHeader *outputHeader;
	DnnHeader inputHeader;
	std::unordered_map<TensorType, IMX500Tensors> offsets;

	/*
	 * Structure of the IMX500 DNN output:
	 * Line [0, ch7_y): Input tensor
	 * Line [ch7_y, ch7_y + ch8_y): Output tensor
	 */

	/*
     * NOTE: this is different form the LIBCAMERA buffer
	 * Structure of the IMX500 DNN output:
	 * Line 0: KPI params
	 * Line [1, x): Input tensor
	 * Line [x, N-1): Output tensor
	 * Line N-1: PQ params
	 */
    
    // Input tensor
    inputHeader = *reinterpret_cast<const DnnHeader *>(buffer);
	if (inputHeader.tensorType != TensorType::InputTensor) {
        std::cout << "[DEBUG] Input tensor is invalid, arborting." << std::endl;
		return {};
	}

    offsets[TensorType::InputTensor].offset = 0;
	offsets[TensorType::InputTensor].valid = (bool)inputHeader.frameValid;

	// std::cout << "[DEBUG] "
	// 	<< "Found input tensor at offset: " << offsets[TensorType::InputTensor].offset
	// 	<< ", valid: " << offsets[TensorType::InputTensor].valid << std::endl;

    // Output tensor
	
	
	// // NOTE: libcamera has no access to ch7_y so there they go over the buffer until they find another header
    // // We know this will be at buffer + offset where offset = (ch7_x * ch7_y)
    // outputHeader = *reinterpret_cast<const DnnHeader *>(buffer + offset);
    // if (outputHeader.tensorType != TensorType::OutputTensor) {
    //     std::cout << "[DEBUG] Output tensor is invalid, arborting." << std::endl;
	// 	return {};
	// }

    // offsets[TensorType::OutputTensor].offset = offset;
	// offsets[TensorType::OutputTensor].valid = outputHeader.frameValid;
	const uint8_t *src = buffer + TensorStride;

	while (src < buffer + buffer_size) {
		outputHeader = reinterpret_cast<const DnnHeader *>(src);
		if (outputHeader->frameCount == inputHeader.frameCount &&
		    outputHeader->apParamSize == inputHeader.apParamSize &&
		    outputHeader->maxLineLen == inputHeader.maxLineLen &&
		    outputHeader->tensorType == TensorType::OutputTensor) {
			offsets[TensorType::OutputTensor].offset = src - buffer;
			offsets[TensorType::OutputTensor].valid = (bool)outputHeader->frameValid;
			// std::cout << "[DEBUG] "
			// 	<< "Found output tensor at offset: " << offsets[TensorType::OutputTensor].offset
			// 	<< ", valid: " << offsets[TensorType::OutputTensor].valid << std::endl;
			break;
		}
		src += TensorStride;
	}


    return offsets;
}


// -----------------------------------------------------------------------------
//  Entry point
// -----------------------------------------------------------------------------
std::unique_ptr<uint8_t[]> savedInputTensor_;


void parse_inference_data(uint8_t *buffer, size_t buffer_size, RequestInterface *req, RequestPool* rp, bool enable_input_tensor) {

    // NOTE util only required for the ch7_x & ch7_y, ch8_x & ch8_y values !!
    // Consider finding a way to get rid of it
    // size_t output_tensor_offset = util.mFPKinfo.dnn[0].dd_ch7_x * util.mFPKinfo.dnn[0].dd_ch7_y;
    
	std::unordered_map<TensorType, IMX500Tensors> offsets = get_tensor_offsets(buffer, buffer_size);
    auto itIn = offsets.find(TensorType::InputTensor);
	auto itOut = offsets.find(TensorType::OutputTensor);

	// Reset the data sizes
	req->input_tensor.data_size = 0;
	req->output_tensor.data_size = 0;

    // Parse input tensor
	if (enable_input_tensor) {
    	if (itIn != offsets.end() && itOut != offsets.end()) { // both input & output tensors headers found
			const unsigned int inputTensorOffset = itIn->second.offset;
			const unsigned int outputTensorOffset = itOut->second.offset;
			const unsigned int inputTensorSize = outputTensorOffset - inputTensorOffset;
			uint8_t* inputTensor = nullptr;
			
			if (itIn->second.valid) {
				if (itOut->second.valid) {
					/* Valid input and output tensor, get the span directly from the current cache. */
					inputTensor = buffer + inputTensorOffset;
				} else {
					/*
					* Invalid output tensor with valid input tensor.
					* This is likely because the DNN takes longer than
					* a frame time to generate the output tensor.
					*
					* In such cases, we don't process the input tensor,
					* but simply save it for when the next output
					* tensor is valid. This way, we ensure that both
					* valid input and output tensors are in lock-step.
					*/
					savedInputTensor_ = std::make_unique<uint8_t[]>(inputTensorSize);
					memcpy(savedInputTensor_.get(), buffer + inputTensorOffset,
						inputTensorSize);
				}
			} else if (itOut->second.valid && savedInputTensor_) {
				/*
				* Invalid input tensor with valid output tensor. This is
				* likely because the DNN takes longer than a frame time
				* to generate the output tensor.
				*
				* In such cases, use the previously saved input tensor
				* if possible.
				*/
				inputTensor = savedInputTensor_.get();
			}

			if (inputTensor != nullptr)  {
				IMX500InputTensorInfo inputTensorInfo;
				if (!imx500ParseInputTensor(inputTensorInfo, inputTensor)) {

					// Sanity check
					uint32_t data_size = inputTensorInfo.size * sizeof(uint8_t);
					if (data_size != rp->input_tensor_size) {
						std::cerr << "[ERROR] inputTensorInfo.size != rp->input_tensor_size" << std::endl;
					}
					
					// Parsing input tensor successful
					// Fill the request interface
					req->input_tensor.width = inputTensorInfo.width;
					req->input_tensor.height = inputTensorInfo.height;
					req->input_tensor.num_channels = inputTensorInfo.channels;
					req->input_tensor.data_offset = req_get_input_tensor_offset(rp, req->idx);
					req->input_tensor.data_size = inputTensorInfo.size;
					
					memcpy(req_get_input_tensor_ptr(rp, req->idx), inputTensorInfo.data.get(), data_size);
				}
				/* We can now safely clear the saved input tensor. */
				savedInputTensor_.reset();
			}
    	}
	}

    // Parse output tensor
    if (itOut != offsets.end() && itOut->second.valid) {  // output tensor header found and valid
        // TODO: parse the output tensor
        unsigned int outputTensorOffset = itOut->second.offset;
        
        IMX500OutputTensorInfo outputTensorInfo;
		if (!imx500ParseOutputTensor(outputTensorInfo, buffer + outputTensorOffset)) {

            // Parsing output tensor successful
			// Fill the request interface
			
            if (outputTensorInfo.numTensors < MAX_NUM_TENSORS) {
                req->output_tensor.num_tensors = outputTensorInfo.numTensors;

                for (unsigned int i = 0; i < outputTensorInfo.numTensors; i++) {
					req->output_tensor.info[i].tensor_data_num = outputTensorInfo.tensorDataNum[i];
					req->output_tensor.info[i].num_dimensions = outputTensorInfo.numDimensions[i];
					for (unsigned int j = 0; j < outputTensorInfo.numDimensions[i]; j++)
						req->output_tensor.info[i].size[j] = outputTensorInfo.vecDim[i][j].size;
				}

            } else {
				std::cout << "[DEBUG] "
					<< "IMX500 output tensor info export failed, numTensors > MAX_NUM_TENSORS" << std::endl;
			}

			// Sanity check
			uint32_t data_size = outputTensorInfo.totalSize * sizeof(float);
			if (data_size != rp->output_tensor_size) {
				std::cerr << "[ERROR] outputTensorInfo.totalSize != rp->output_tensor_size" << std::endl;
			}

			req->output_tensor.data_offset = req_get_output_tensor_offset(rp, req->idx);
			req->output_tensor.data_size = data_size;

			memcpy(req_get_output_tensor_ptr(rp, req->idx), outputTensorInfo.data.get(), data_size);
        }
    }
}