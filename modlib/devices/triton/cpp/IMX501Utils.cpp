// =============================================================================
//
//  Copyright (c) 2023, Lucid Vision Labs, Inc.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//
// =============================================================================

// Includes --------------------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <exception>
#include "ArenaApi.h"
#include "IMX501Utils.h"
using std::cout;
using std::endl;
// Macros ----------------------------------------------------------------------
#define FPK_INFO_SIGNATURE      0x4443554C  // 'L' 'U' 'C' 'D'
#define FPK_INFO_CHIP_ID        501         // IMX501
#define FPK_INFO_DATA_TYPE      0x0001      // DNN Output setting

#define SENSOR_W  4052
#define SENSOR_H  3036


// Namespace -------------------------------------------------------------------
namespace ArenaExample {

// ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//  IMX501Utils class
// ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
// Constructors and Destructor -------------------------------------------------
// -----------------------------------------------------------------------------
//  IMX501Utils
// -----------------------------------------------------------------------------
IMX501Utils::IMX501Utils(Arena::IDevice *inDevice, bool inVerboseMode)
{
  mVerboseMode = inVerboseMode;

  mDevice = inDevice;

  mNodeMap = nullptr;
  mNodeMap = mDevice->GetNodeMap();
  if (mNodeMap == nullptr)
    throw std::runtime_error("Couldn't get the NodeMap");

  memset(&mFPKinfo, 0, sizeof(mFPKinfo));
}

// -----------------------------------------------------------------------------
//  ~IMX501Utils
// -----------------------------------------------------------------------------
IMX501Utils::~IMX501Utils()
{
}

// -----------------------------------------------------------------------------
// Public member functions -----------------------------------------------------
// -----------------------------------------------------------------------------
//  InitCameraToOutputDNN
// -----------------------------------------------------------------------------
void IMX501Utils::InitCameraToOutputDNN(bool inEnableNoRawOutput, bool inEnableSensorISPManualMode)
{

  if (mDevice == nullptr)
    throw std::runtime_error("mDevice == nullptr");

  GenApi::CBooleanPtr pNetworkEnable = mNodeMap->GetNode("DeepNeuralNetworkEnable");
  if (pNetworkEnable == nullptr)
    throw std::runtime_error("Couldn't find node: DeepNeuralNetworkEnable");

  if (pNetworkEnable->GetAccessMode() == GenApi::EAccessMode::RO)
  {
    // Always output the log for now.
    // Since the execution of DeepNeuralNetworkLoad takes time 
    printf("Execute: DeepNeuralNetworkLoad\n");
    ExecuteNode(mNodeMap, "DeepNeuralNetworkLoad");
    printf("Execute: Done\n");
  }

  GenApi::CBooleanPtr pManualEnable = mNodeMap->GetNode("DeepNeuralNetworkISPAutoEnable");
  if (pManualEnable == nullptr)
    throw std::runtime_error("Couldn't find node: DeepNeuralNetworkISPAutoEnable");
  if (pManualEnable->GetValue() != (!inEnableSensorISPManualMode))
  {
    if (pNetworkEnable->GetValue() != false)
      pNetworkEnable->SetValue(false);
    pManualEnable->SetValue((!inEnableSensorISPManualMode));
    if (mVerboseMode)
      printf("Changed: DeepNeuralNetworkISPAutoEnable\n");
  }

  if (pNetworkEnable->GetValue() == false)
  {
    if (mVerboseMode)
      printf("Enable: DeepNeuralNetworkEnable\n");
    pNetworkEnable->SetValue(true);
    if (mVerboseMode)
      printf("Enable: Done\n");
  }

  SetBooleanNode(mNodeMap, "GammaEnable", true);
  SetFloatNode(mNodeMap, "Gamma", 0.45);

  RetrieveFPKinfo(mNodeMap, &mFPKinfo);
  if (mVerboseMode)
    DumpFPKinfo(&mFPKinfo);
  if (ValidateFPKinfo(&mFPKinfo) == false)
    throw std::runtime_error("Received invalid fpk_info");

  if (inEnableNoRawOutput)
  {
    std::cout << "Setting image size to 4x4" << std::endl;
    SetIntNode(mNodeMap, "Width",  4);
    SetIntNode(mNodeMap, "Height", 4);
    mNoRawOutput = true;
  } else {
    mNoRawOutput = false;
  }
}

// // -----------------------------------------------------------------------------
// //  SuppressRawOutput
// // -----------------------------------------------------------------------------
// const void IMX501Utils::SuppressRawOutput(bool inEnableNoRawOutput) {
//   if (inEnableNoRawOutput)
//   {
//     SetImageSize(4, 4);
//     mNoRawOutput = true;
//   }else{
//     mNoRawOutput = false;
//     SetImageOffset();
//     SetImageSize();
//   }
// }

// -----------------------------------------------------------------------------
//  SetCameraDefaults
// -----------------------------------------------------------------------------
const void IMX501Utils::SetCameraDefaults(){
  // Set binning to 1
  GenApi::CIntegerPtr pNode;
  GenApi::CEnumerationPtr pEnumeration = mNodeMap->GetNode("BinningSelector");
  if (pEnumeration == nullptr)
    throw std::runtime_error("Couldn't find the Enumeration node");
  GenApi::CEnumEntryPtr pEntry = pEnumeration->GetEntryByName("Sensor");
  if (pEntry == nullptr)
    throw std::runtime_error("Couldn't find the Enumeration entry");
  pEnumeration->SetIntValue(pEntry->GetValue());
  pNode = mNodeMap->GetNode("BinningHorizontal");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the BinningHorizontal node");
  pNode->SetValue(1);

  // Set Sensor image size to 4052x3036
  pNode = mNodeMap->GetNode("OffsetX");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the OffsetX node");
  pNode->SetValue(0); // Please note that we need to set OffsetX to 0 first to prevent the ROI setting violation
  pNode = mNodeMap->GetNode("OffsetY");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the OffsetY node");
  pNode->SetValue(0); // Please note that we need to set OffsetY to 0 first to prevent the ROI setting violation
  pNode = mNodeMap->GetNode("Width");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the Width node");
  pNode->SetValue(SENSOR_W);
  pNode = mNodeMap->GetNode("Height");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the Height node");
  pNode->SetValue(SENSOR_H);

  // Set framerate to auto
  GenApi::CBooleanPtr pBoolean = mNodeMap->GetNode("AcquisitionFrameRateEnable");
  if (pBoolean == nullptr)
    throw std::runtime_error("Couldn't find the AcquisitionFrameRateEnable node");
  pBoolean->SetValue(false);
}

// -----------------------------------------------------------------------------
//  SetDNNDefaults
// -----------------------------------------------------------------------------
const void IMX501Utils::SetDNNDefaults(){
  // Set Sensor ISP image size to 4052x3036
  GenApi::CIntegerPtr pNode;
  pNode = mNodeMap->GetNode("DeepNeuralNetworkISPOffsetX");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the DeepNeuralNetworkISPOffsetX node");
  pNode->SetValue(0); // Please note that we need to set OffsetX to 0 first to prevent the ROI setting violation
  pNode = mNodeMap->GetNode("DeepNeuralNetworkISPOffsetY");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the DeepNeuralNetworkISPOffsetY node");
  pNode->SetValue(0); // Please note that we need to set OffsetY to 0 first to prevent the ROI setting violation
  pNode = mNodeMap->GetNode("DeepNeuralNetworkISPWidth");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the DeepNeuralNetworkISPWidth node");
  pNode->SetValue(SENSOR_W);
  pNode = mNodeMap->GetNode("DeepNeuralNetworkISPHeight");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the DeepNeuralNetworkISPHeight node");
  pNode->SetValue(SENSOR_H);
}

// -----------------------------------------------------------------------------
//  SetBinning
// -----------------------------------------------------------------------------
const void IMX501Utils::SetBinning(unsigned int scale){
  GenApi::CIntegerPtr pNode;
  // Set up the binning mode -----------------------------------------------------
  GenApi::CEnumerationPtr pEnumeration = mNodeMap->GetNode("BinningSelector");
  if (pEnumeration == nullptr)
    throw std::runtime_error("Couldn't find the Enumeration node");
  GenApi::CEnumEntryPtr pEntry = pEnumeration->GetEntryByName("Sensor");
  if (pEntry == nullptr)
    throw std::runtime_error("Couldn't find the Enumeration entry");
  pEnumeration->SetIntValue(pEntry->GetValue());
  pNode = mNodeMap->GetNode("BinningHorizontal");
  if (pNode == nullptr)
    throw std::runtime_error("Couldn't find the BinningHorizontal node");
  pNode->SetValue(scale);
}

// -----------------------------------------------------------------------------
//  SetFPS
// -----------------------------------------------------------------------------
const void IMX501Utils::SetFPS(double fps){
  GenApi::CBooleanPtr pBoolean = mNodeMap->GetNode("AcquisitionFrameRateEnable"); // boolean node
  if (pBoolean == nullptr)
    throw std::runtime_error("Couldn't find the AcquisitionFrameRateEnable node");
  GenApi::CFloatPtr pFloat = mNodeMap->GetNode("AcquisitionFrameRate"); // float node
  if (pFloat == nullptr)
    throw std::runtime_error("Couldn't find the AcquisitionFrameRate node");
  if (fps <= 0.) {
    cout << "[Info] Auto framerate setting is enabled." << endl;
    pBoolean->SetValue(false);
  }
  else {
    pBoolean->SetValue(true);
    if (fps > pFloat->GetMax()) {
        std::cout << "[INFO] Framerate is set to " << pFloat->GetMax() << " while user input: " << fps << std::endl;
        pFloat->SetValue(pFloat->GetMax());
    }
    else {
        pFloat->SetValue(fps);
    }
  }
}

// -----------------------------------------------------------------------------
//  GetFPS
// -----------------------------------------------------------------------------
const double IMX501Utils::GetFPS() {
  GenApi::CBooleanPtr pBoolean = mNodeMap->GetNode("AcquisitionFrameRateEnable"); // boolean node
  if (pBoolean == nullptr)
    throw std::runtime_error("Couldn't find the AcquisitionFrameRateEnable node");
  GenApi::CFloatPtr pFloat = mNodeMap->GetNode("AcquisitionFrameRate"); // float node
  if (pFloat == nullptr)
    throw std::runtime_error("Couldn't find the AcquisitionFrameRate node");

  return pFloat->GetValue();
  }

// -----------------------------------------------------------------------------
//  SetImageSize
// -----------------------------------------------------------------------------
const void IMX501Utils::SetImageSize(unsigned int width, unsigned int height){
 if (mNoRawOutput == false){
   GenApi::CIntegerPtr pNode = mNodeMap->GetNode("Width");
   if (pNode == nullptr)
     throw std::runtime_error("Couldn't find the Width node");

   if (width > pNode->GetMax()) {
     std::cout << "[INFO] Width is set to " << pNode->GetMax() << " while user input: " << width << std::endl;
       pNode->SetValue(pNode->GetMax());
   }
   else {
       pNode->SetValue(width);
   }

   pNode = mNodeMap->GetNode("Height");
   if (pNode == nullptr)
     throw std::runtime_error("Couldn't find the Height node");
   if (height > pNode->GetMax()) {
     std::cout << "[INFO] Height is set to " << pNode->GetMax() << " while user input: " << height << std::endl;
       pNode->SetValue(pNode->GetMax());
   }
   else {
       pNode->SetValue(height);
   }
 }
 else{
   std::cout << "[INFO] Output size is limited to 4x4 when isNoRawOutput = true" << std::endl;
 }
}

// -----------------------------------------------------------------------------
//  SetImageOffset
// -----------------------------------------------------------------------------
const void IMX501Utils::SetImageOffset(unsigned int x, unsigned int y) {
   GenApi::CIntegerPtr pNode = mNodeMap->GetNode("OffsetX");
   if (pNode == nullptr)
       throw std::runtime_error("Couldn't find the OffsetX node");

   if (x > pNode->GetMax()) {
     std::cout << "[INFO] OffsetX is set to " << pNode->GetMax() << " while user input: " << x << std::endl;
       pNode->SetValue(pNode->GetMax());
   }
   else {
       pNode->SetValue(x);
   }

   pNode = mNodeMap->GetNode("OffsetY");
   if (pNode == nullptr)
       throw std::runtime_error("Couldn't find the OffsetY node");
   if ( y > pNode->GetMax()) {
     std::cout << "[INFO] OffsetY is set to " << pNode->GetMax() << " while user input: " << y << std::endl;
       pNode->SetValue(pNode->GetMax());
   }
   else {
       pNode->SetValue(y);
   }
}

// -----------------------------------------------------------------------------
//  SetDNNImageSize
// -----------------------------------------------------------------------------
const void IMX501Utils::SetDNNImageSize(unsigned int width, unsigned int height) {
 GenApi::CIntegerPtr pNode = mNodeMap->GetNode("DeepNeuralNetworkISPWidth");
 if (pNode == nullptr)
     throw std::runtime_error("Couldn't find the DeepNeuralNetworkISPWidth node");

 if (width > pNode->GetMax()) {
   std::cout << "[INFO] DeepNeuralNetworkISPWidth is set to " << pNode->GetMax() << " while user input: " << width << std::endl;
     pNode->SetValue(pNode->GetMax());
 }
 else {
     pNode->SetValue(width);
 }

 pNode = mNodeMap->GetNode("DeepNeuralNetworkISPHeight");
 if (pNode == nullptr)
     throw std::runtime_error("Couldn't find the DeepNeuralNetworkISPHeight node");
 if (height > pNode->GetMax()) {
   std::cout << "[INFO] DeepNeuralNetworkISPHeight is set to " << pNode->GetMax() << " while user input: " << height << std::endl;
     pNode->SetValue(pNode->GetMax());
 }
 else {
     pNode->SetValue(height);
 }
}

// -----------------------------------------------------------------------------
//  SetDNNImageOffset
// -----------------------------------------------------------------------------
const void IMX501Utils::SetDNNImageOffset(unsigned int x, unsigned int y) {
 GenApi::CIntegerPtr pNode = mNodeMap->GetNode("DeepNeuralNetworkISPOffsetX");
 if (pNode == nullptr)
     throw std::runtime_error("Couldn't find the DeepNeuralNetworkISPOffsetX node");

 if (x > pNode->GetMax()) {
   std::cout << "[INFO] DeepNeuralNetworkISPOffsetX is set to " << pNode->GetMax() << " while user input: " << x << std::endl;
     pNode->SetValue(pNode->GetMax());
 }
 else {
     pNode->SetValue(x);
 }

 pNode = mNodeMap->GetNode("DeepNeuralNetworkISPOffsetY");
 if (pNode == nullptr)
     throw std::runtime_error("Couldn't find the DeepNeuralNetworkISPOffsetY node");
 if ( y > pNode->GetMax()) {
   std::cout << "[INFO] DeepNeuralNetworkISPOffsetY is set to " << pNode->GetMax() << " while user input: " << y << std::endl;
     pNode->SetValue(pNode->GetMax());
 }
 else {
     pNode->SetValue(y);
 }
}





// -----------------------------------------------------------------------------
//  IsVerboseMode
// -----------------------------------------------------------------------------
bool IMX501Utils::IsVerboseMode()
{
  return mVerboseMode;
}

// -----------------------------------------------------------------------------
// Public static member functions -------------------------------------------
// -----------------------------------------------------------------------------
//  UploadFileToCamera
// -----------------------------------------------------------------------------
void IMX501Utils::UploadFileToCamera(Arena::IDevice* inDevice,
  CameraFileType inFileType, const void* inData, size_t inDataSize,
  int (*inProgressFunc)(size_t, size_t))
{
  const char* fileName = "";
  switch (inFileType)
  {
    case CameraFileType::FILE_DEEP_NEURAL_NETWORK_FIRMWARE:
      fileName = "DeepNeuralNetworkFirmware";
      break;
    case CameraFileType::FILE_DEEP_NEURAL_NETWORK_LOADER:
      fileName = "DeepNeuralNetworkLoader";
      break;
    case CameraFileType::FILE_DEEP_NEURAL_NETWORK_NETWORK:
      fileName = "DeepNeuralNetworkNetwork";
      break;
    case CameraFileType::FILE_DEEP_NEURAL_NETWORK_INFO:
      fileName = "DeepNeuralNetworkInfo";
      break;
    case CameraFileType::FILE_DEEP_NEURAL_NETWORK_CLASSIFICATION:
      fileName = "DeepNeuralNetworkClassification";
      break;
    default:
      throw std::runtime_error("inFileType == Unknown");
      break;
  }
  GenApi::INodeMap* pNodeMap = inDevice->GetNodeMap();

  WriteFile(pNodeMap, fileName, inData, inDataSize, 0, inProgressFunc);
}

// -----------------------------------------------------------------------------
//  ReloadNetwork
// -----------------------------------------------------------------------------
void IMX501Utils::ReloadNetwork(Arena::IDevice* inDevice)
{
  GenApi::INodeMap *nodeMap = inDevice->GetNodeMap();
  if (nodeMap == nullptr)
    throw std::runtime_error("Couldn't get the NodeMap");

  GenApi::CBooleanPtr pNetworkEnable = nodeMap->GetNode("DeepNeuralNetworkEnable");
  if (pNetworkEnable == nullptr)
    throw std::runtime_error("Couldn't find node: DeepNeuralNetworkEnable");

  if (pNetworkEnable->GetValue() != false)
    pNetworkEnable->SetValue(false);

  // Always output the log for now.
  // Since the execution of DeepNeuralNetworkLoad takes time 
  printf("Execute: DeepNeuralNetworkLoad\n");
  ExecuteNode(nodeMap, "DeepNeuralNetworkLoad");
  printf("Execute: Done\n");

  pNetworkEnable->SetValue(true);
}

// -----------------------------------------------------------------------------
// Protected static member functions -------------------------------------------
// -----------------------------------------------------------------------------
//  RetrieveFPKinfo
// -----------------------------------------------------------------------------
void IMX501Utils::RetrieveFPKinfo(GenApi::INodeMap *inNodeMap,
                              fpk_info *outInfo)
{
  ReadFile(inNodeMap, "DeepNeuralNetworkInfo",
               outInfo, sizeof(fpk_info));
}

// -----------------------------------------------------------------------------
//  RetrieveLabelData
// -----------------------------------------------------------------------------
uint8_t *IMX501Utils::RetrieveLabelData(GenApi::INodeMap *inNodeMap, size_t *outDataSize)
{
  *outDataSize = (size_t )GetFileSize(inNodeMap, "DeepNeuralNetworkClassification");
  if (*outDataSize == 0)
    return NULL;
  uint8_t *dataBuf = new uint8_t[*outDataSize];
  if (dataBuf == NULL)
    return NULL;
  ReadFile(inNodeMap, "DeepNeuralNetworkClassification",
               dataBuf, *outDataSize);
  return dataBuf;
}

// -----------------------------------------------------------------------------
//  MakeLabelList
// -----------------------------------------------------------------------------
void IMX501Utils::MakeLabelList(const uint8_t *inLabelData, size_t inDataSize,
                                  std::vector<std::string> *outList)
{
  if (inLabelData == NULL || inDataSize == 0 || outList == NULL)
    return;

  outList->clear();

  size_t  labelNum = 0;
  size_t  i, from = 0;
  for (i = 0; i < inDataSize; i++)
  {
    if (inLabelData[i] == 0x0A) // LF
    {
      if (from == i) // no char between two LF
        outList->push_back("");
      else if (inLabelData[from-1] == 0x0D) { // CR
          outList->push_back(std::string((const char*)&(inLabelData[from]), i - from - 1));
        }
        else {
          outList->push_back(std::string((const char*)&(inLabelData[from]), i - from));
        }

      from = i + 1;
    }
  }
  if (from != i) // if LF not found in EOF
    outList->push_back(std::string((const char*)&(inLabelData[from]), i - from));
}

// -----------------------------------------------------------------------------
//  ValidateFPKinfo
// -----------------------------------------------------------------------------
bool IMX501Utils::ValidateFPKinfo(const fpk_info *inInfo)
{
  // check the header consistency
  if (inInfo->signature      != FPK_INFO_SIGNATURE ||
      inInfo->version_major  >  0x02 ||
      inInfo->chip_id        != FPK_INFO_CHIP_ID   ||
      inInfo->data_type      != FPK_INFO_DATA_TYPE)
    return false;

  // sanity check
  if (inInfo->dnn[0].dd_ch7_x  == 0 ||
      inInfo->dnn[0].dd_ch7_y  == 0 ||
      inInfo->dnn[0].dd_ch8_x  == 0 ||
      inInfo->dnn[0].dd_ch8_y  == 0 )
    return false;

  // the current version doesn't support the following configuration
  if (inInfo->dnn[0].dd_ch7_x != inInfo->dnn[0].dd_ch8_x)
    return false;

  return true;
}

// -----------------------------------------------------------------------------
//  DumpFPKinfo
// -----------------------------------------------------------------------------
void IMX501Utils::DumpFPKinfo(const fpk_info *inInfo)
{
  printf("[fpk_info]\n");
  printf("signature:     0x%08X\n", inInfo->signature);
  printf("data_size:     %d\n", inInfo->data_size);
  printf("version_major: %d\n", (int )inInfo->version_major);
  printf("version_minor: %d\n", (int )inInfo->version_minor);
  printf("chip_id:       %d\n", (int )inInfo->chip_id);
  printf("data_type:     %d\n", (int )inInfo->data_type);

  char  buf[65];
  memcpy(buf, inInfo->fpk_info_str, 64);
  buf[64] = 0;
  if (strlen(buf) > 63)
    printf("warning: fpk_info_str is not null terminated\n");
  printf("fpk_info_str:  %s\n", buf);
  printf("network_num:   %d\n", inInfo->network_num);

  printf("dnn[0]\n");
  printf("dd_ch7_x:      %d\n", (int )inInfo->dnn[0].dd_ch7_x);
  printf("dd_ch7_y:      %d\n", (int )inInfo->dnn[0].dd_ch7_y);
  printf("dd_ch8_x:      %d\n", (int )inInfo->dnn[0].dd_ch8_x);
  printf("dd_ch8_y:      %d\n", (int )inInfo->dnn[0].dd_ch8_y);

  for (int i = 0; i < 8; i++)
    printf("input_tensor_norm_k[%d]: 0x%04X\n", i, (int )inInfo->dnn[0].input_tensor_norm_k[i]);
  printf("input_tensor_format:     %d\n", (int )inInfo->dnn[0].input_tensor_format);
  printf("input_tensor_norm_ygain: 0x%04X\n", (int )inInfo->dnn[0].input_tensor_norm_ygain);
  printf("input_tensor_norm_yadd:  0x%04X\n", (int )inInfo->dnn[0].input_tensor_norm_yadd);
  printf("y_clip:        0x%08X\n", inInfo->dnn[0].y_clip);
  printf("cb_clip:       0x%08X\n", inInfo->dnn[0].cb_clip);
  printf("cr_clip:       0x%08X\n", inInfo->dnn[0].cr_clip);
  for (int i = 0; i < 4; i++)
  {
    printf("input_norm[%d]:         0x%04X\n", i, (int )inInfo->dnn[0].input_norm[i]);
    printf("input_norm_shift[%d]:   0x%02X\n", i, (int )inInfo->dnn[0].input_norm_shift[i]);
    printf("input_norm_clip[%d]:    0x%08X\n", i, inInfo->dnn[0].input_norm_clip[i]);
  }
  printf("\n");
}

// -----------------------------------------------------------------------------
//  ReadFile
// -----------------------------------------------------------------------------
void IMX501Utils::ReadFile(GenApi::INodeMap *inNodeMap,
                              const char *inFileName,
                              void *inFileBuf, size_t inFileSize,
                              int inOffset)
{
  SetEnumNode(inNodeMap, "FileSelector", inFileName);
  SetEnumNode(inNodeMap, "FileOperationSelector", "Open");
  SetEnumNode(inNodeMap, "FileOpenMode", "Read");
  ExecuteNode(inNodeMap, "FileOperationExecute");

  SetEnumNode(inNodeMap, "FileOperationSelector", "Read");
  SetIntNode(inNodeMap, "FileAccessOffset", inOffset);
  SetIntNode(inNodeMap, "FileAccessLength", (int64_t )inFileSize);
  ExecuteNode(inNodeMap, "FileOperationExecute");

  GenApi::CRegisterPtr pRegister = inNodeMap->GetNode("FileAccessBuffer");
  pRegister->Get((uint8_t *)inFileBuf, inFileSize);

  SetEnumNode(inNodeMap, "FileOperationSelector", "Close");
  ExecuteNode(inNodeMap, "FileOperationExecute");
}

// -----------------------------------------------------------------------------
//  GetFileSize
// -----------------------------------------------------------------------------
int64_t IMX501Utils::GetFileSize(GenApi::INodeMap *inNodeMap,
                                  const char *inFileName)
{
  SetEnumNode(inNodeMap, "FileSelector", inFileName);
  return GetIntNode(inNodeMap, "FileSize");
}

// -----------------------------------------------------------------------------
//  WriteFile
// -----------------------------------------------------------------------------
void IMX501Utils::WriteFile(GenApi::INodeMap* inNodeMap,
  const char* inFileName,
  const void* inFileBuf, size_t inFileSize,
  int inOffset,
  int (*inProgressFunc)(size_t, size_t))
{
  SetEnumNode(inNodeMap, "FileSelector", inFileName);
  SetEnumNode(inNodeMap, "FileOperationSelector", "Open");
  SetEnumNode(inNodeMap, "FileOpenMode", "Write");
  ExecuteNode(inNodeMap, "FileOperationExecute");

  GenApi::CRegisterPtr pRegister = inNodeMap->GetNode("FileAccessBuffer");
  size_t buf_length = pRegister->GetLength();
  SetEnumNode(inNodeMap, "FileOperationSelector", "Write");

  size_t  remainingSize = inFileSize;
  size_t  offset = inOffset;
  const uint8_t* dataPtr = (uint8_t*)inFileBuf;
  while (remainingSize != 0)
  {
    size_t upload_size = remainingSize;
    if (upload_size > buf_length)
      upload_size = buf_length;

    SetIntNode(inNodeMap, "FileAccessOffset", offset);
    SetIntNode(inNodeMap, "FileAccessLength", (int64_t)upload_size);
    pRegister->Set(dataPtr, upload_size);
    ExecuteNode(inNodeMap, "FileOperationExecute");
    //
    remainingSize -= upload_size;
    offset += upload_size;
    dataPtr += upload_size;
    if (inProgressFunc != nullptr)
    {
      int result;
      // result = 0 : no action
      // result = 1 : terminate the file write 
      result = inProgressFunc(inFileSize, inFileSize - remainingSize);
      if (result != 0)
        break;
    }
  }

  SetEnumNode(inNodeMap, "FileOperationSelector", "Close");
  ExecuteNode(inNodeMap, "FileOperationExecute");
}

// -----------------------------------------------------------------------------
//  SetEnumNode
// -----------------------------------------------------------------------------
void IMX501Utils::SetEnumNode(GenApi::INodeMap *inNodeMap,
                              const char *inNodeName, const char *inValue)
{
  GenApi::CEnumerationPtr pEnumeration = inNodeMap->GetNode(inNodeName);
  if (pEnumeration == nullptr)
    throw std::runtime_error("Couldn't find the Enumeration node");
  GenApi::CEnumEntryPtr pEntry = pEnumeration->GetEntryByName(inValue);
  if (pEntry == nullptr)
    throw std::runtime_error("Couldn't find the Enumeration entry");
  pEnumeration->SetIntValue(pEntry->GetValue());
}

// -----------------------------------------------------------------------------
//  SetFloatNode
// -----------------------------------------------------------------------------
void IMX501Utils::SetFloatNode(GenApi::INodeMap* inNodeMap,
                                const char* inNodeName, double inValue)
{
  GenApi::CFloatPtr pFloat = inNodeMap->GetNode(inNodeName);
  if (pFloat == nullptr)
    throw std::runtime_error("Couldn't find the Float node");
  pFloat->SetValue(inValue);
}

// -----------------------------------------------------------------------------
//  SetBooleanNode
// -----------------------------------------------------------------------------
void IMX501Utils::SetBooleanNode(GenApi::INodeMap* inNodeMap,
                                  const char* inNodeName, bool inValue)
{
  GenApi::CBooleanPtr pBoolean = inNodeMap->GetNode(inNodeName);
  if (pBoolean == nullptr)
    throw std::runtime_error("Couldn't find the Boolean node");
  pBoolean->SetValue(inValue);
}

// -----------------------------------------------------------------------------
//  SetIntNode
// -----------------------------------------------------------------------------
void IMX501Utils::SetIntNode(GenApi::INodeMap *inNodeMap,
                              const char *inNodeName, int64_t inValue)
{
  GenApi::CIntegerPtr pInteger = inNodeMap->GetNode(inNodeName);
  if (pInteger == nullptr)
    throw std::runtime_error("Couldn't find the Integer node");
  pInteger->SetValue(inValue);
}

// -----------------------------------------------------------------------------
//  GetIntNode
// -----------------------------------------------------------------------------
int64_t IMX501Utils::GetIntNode(GenApi::INodeMap *inNodeMap,
                              const char *inNodeName)
{
  GenApi::CIntegerPtr pInteger = inNodeMap->GetNode(inNodeName);
  if (pInteger == nullptr)
    throw std::runtime_error("Couldn't find the Integer node");
  return pInteger->GetValue();
}

// -----------------------------------------------------------------------------
//  ExecuteNode
// -----------------------------------------------------------------------------
void IMX501Utils::ExecuteNode(GenApi::INodeMap *inNodeMap,
                              const char *inNodeName)
{
  GenApi::CCommandPtr pCommand = inNodeMap->GetNode(inNodeName);
  if (pCommand == nullptr)
    throw std::runtime_error("Couldn't find the Command node");
  pCommand->Execute();
}

// Namespace -------------------------------------------------------------------
};
