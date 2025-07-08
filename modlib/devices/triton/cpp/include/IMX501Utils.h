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
#ifndef ARENA_EXAMPLE_IMX501_UTILS_H
#define ARENA_EXAMPLE_IMX501_UTILS_H

// Includes --------------------------------------------------------------------
#include <vector>
#include <iostream>
#include <string>
#include <exception>
#include <stdexcept>
#include "ArenaApi.h"
#include "apParams.flatbuffers_generated.h"


// Namespace -------------------------------------------------------------------
namespace ArenaExample {

// ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
//  ArenaExample::IMX501Utils class
// ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
class IMX501Utils
{
public:
  // typedefs ------------------------------------------------------------------

  // Enum ----------------------------------------------------------------------

  /**
  * This enum represents the type of a file uploaded to a camera.
  */
  enum class CameraFileType
  {
    FILE_DEEP_NEURAL_NETWORK_FIRMWARE = 0,    /*!< 0:Deep Neural Network Firmware       = firmware.fpk */
    FILE_DEEP_NEURAL_NETWORK_LOADER,          /*!< 1:Deep Neural Network Loader         = loader.fpk */
    FILE_DEEP_NEURAL_NETWORK_NETWORK,         /*!< 2:Deep Neural Network Network        = network.fpk */
    FILE_DEEP_NEURAL_NETWORK_INFO,            /*!< 3:Deep Neural Network info           = fpk_info.dat */
    FILE_DEEP_NEURAL_NETWORK_CLASSIFICATION,  /*!< 4:Deep Neural Network Classfication  = label.txt */
    CAMERA_FILE_TYPE_UNKNOWN = 0xFFFF         /*!< 0xFFFF: Unknown Camera File Type*/
  };

  // Constructors and Destructor -----------------------------------------------
  /**
  * @fn IMX501Utils(Arena::IDevice *inDevice, bool inVerboseMode = false)
  *
  * @param inDevice
  *   - Type: Arena::IDevice*
  *   - Pointer to the IDevice object of ArenaSDK
  * @param inVerboseMode
  *   - Type: bool
  *   - Default: false
  *   - Set true to enable a verbose mode. In the verbose mode, the created
  *     object will output additional information to a console.
  *
  * The <B> constructor </B> builds a utility object for the specified camera.
  * The verbose mode can be enable by specifying true to inVerboseMode.
  */
  IMX501Utils(Arena::IDevice *inDevice, bool inVerboseMode = false);

  /**
  * @fn virtual ~IMX501Utils()
  *
  * A destructor
  */
  virtual ~IMX501Utils();

  // Member functions ----------------------------------------------------------
  /**
  * @fn void  InitCameraToOutputDNN(bool inEnableNoRawOutput, bool inEnableSensorISPManualMode)
  *
  * @param inEnableNoRawOutput
  *   - Type: bool
  *   - Enable the no raw output streaming mode by specifying true
  *
  * @param inEnableSensorISPManualMode
  *   - Type: bool
  *   - Enable the sensor ISP manual mode by specifying true
  *
  * @return
  *   - none
  *
  * <B> InitCameraToOutputDNN </B> initializes the specified camera
  * with necessary settings for running DNN inference on IMX501,
  * downloads a label text file and a fpk_info.dat from the camera
  * and prepares internal buffers.
  * If the network is not loaded into the IMX501, this function
  * instructs the camera to uplaod the network into the sensor
  * and enable it. This process requires about 30 sec to
  * finish uploading and happens only once after powering
  * up the camera.
  * The function needs to be called once after creating the IMX501Utils
  * object for the initialization.
  * Wether to enable the no raw output streaming mode or the sensor
  * ISP manual mode or not can be specified as arguments.
  * These input parameters are also optional.
  */
  void  InitCameraToOutputDNN(bool inEnableNoRawOutput = false, bool inEnableSensorISPManualMode = false);

  // // ---------------------------------------------------------------------------
  // /**
  // * @fn void  SuppressRawOutput(bool inEnableNoRawPutput)
  // *
  // * <B> SuppressRawOutput </B> enables/disables raw output for original image.
  // * When true is passed, the output size is set to 4x4 (System minimum); 
  // * Otherwise, the output size is reset to its current maximum.
  // * Range: 1-8
  // */
  // const void SuppressRawOutput(bool inEnableNoRawOutput=true);

  /**
  * @fn void  SetCameraDefaults
  *
  * <B> SetCameraDefaults </B> sets initial camera related setting to the camera.
  * This function resets binning, ROI and framerate settings.
  */
  const void SetCameraDefaults();

  /**
  * @fn void  SetDNNDefaults
  *
  * <B> SetDNNDefaults </B> sets initial DNN related setting to the camera.
  * This function activates DNN and resets DNN ROI setting.
  */
  const void SetDNNDefaults();

  // ---------------------------------------------------------------------------
  /**
  * @fn void  SetBinning(unsigned int scale)
  *
  * <B> SetBinning </B> sets a designated Binning to a camera.
  * Range: 1-8
  */
  const void SetBinning(unsigned int scale = 1);

  // ---------------------------------------------------------------------------
  /**
  * @fn void  SetFPS(double fps)
  *
  * <B> SetFPS </B> sets a designated framerate to a camera.
  * If 0 or less is set, camera framerate will be automatically controlled.
  * If input value exceeds system limitation, max available value
  * will be set.
  */
  const void SetFPS(double fps = -1.);

  // ---------------------------------------------------------------------------
/**
* @fn double  GetFPS()
*
* <B> GetFPS </B> gets a current framerate from a camera.
* If frame rate is auto, returns negative value.
*/
  const double GetFPS();
 // ---------------------------------------------------------------------------
 /**
 * @fn void  SetImageSize(unsigned int width, unsigned int height)
 *
 * <B> SetImageSize </B> sets a designated camera image width and height.
 * The value must be specified with pixels after binning.
 * If input value exceeds system limitation, max available value
 * will be set.
 * This function is invalid when isNoRawOutput = true.
 */
 const void SetImageSize(unsigned int width = 4052, unsigned int height = 3036);

 // ---------------------------------------------------------------------------
/**
* @fn void  SetImageOffset(unsigned int x, unsigned int y)
*
* <B> SetImageOffset </B> sets a designated camera image offsets.
* The value must be specified with pixels after binning.
* If input value exceeds system limitation, max available value
* will be set.
*/
 const void SetImageOffset(unsigned int x = 0, unsigned int y = 0);
 // ---------------------------------------------------------------------------
/**
* @fn void  SetDNNImageSize(unsigned int width, unsigned int height)
*
* <B> SetDNNImageSize </B> sets a designated DNN ROI width and height.
* The value must be specified with 4052x3036 scale.
* If input value exceeds system limitation, max available value
* will be set.
*/
 const void SetDNNImageSize(unsigned int width = 4052, unsigned int height = 3036);

 // ---------------------------------------------------------------------------
/**
* @fn void  SetDNNImageOffset(unsigned int x, unsigned int y)
*
* <B> SetDNNImageOffset </B> sets a designated DNN ROI offsets.
* The value must be specified with 4052x3036 scale.
* If input value exceeds system limitation, max available value
* will be set.
*/
const void SetDNNImageOffset(unsigned int x = 0, unsigned int y = 0);


/**
  * @fn bool  IsVerboseMode()
  *
  * @return
  *   - True if the object is in the verbose mode
  *   - Otherwise, false
  *
  * <B> IsVerboseMode </B> returns the status of the verbose mode.
  * If true, the object is in the verbose mode.
  */
  bool  IsVerboseMode();

  /**
  * @fn bool  UploadFileToCamera(Arena::IDevice* inDevice, CameraFileType inFileType, void* inData, size_t inDataSize)
  *
  * @param inDevice
  *   - Type: Arena::IDevice*
  *   - Pointer to the IDevice object of ArenaSDK
  * @param inFileType
  *   - Type: CameraFileType
  *   - Specify the camera file type to upload to the camera
  * @param inData
  *   - Type: void*
  *   - Pointer to the data buffer for uploading
  * @param inDataSize
  *   - Type: size_t
  *   - size of data
  * @param inProgressFunc
  *   - Type: int (*inProgressFunc)(size_t inFileSize, size_t inFileTransferred)
  *   - Pointer to the optional callback function
  *
  * The <B> UploadFileToCamera </B> uploads data to the specified camera.
  * Uploaded data will be stored as a file in the camera.
  * Can specify a callback function to inform the progress of the file upload process to the caller (optional).
  * inFileSize is the size of the file (same as inDataSize)
  * inFileTransferred is the size of the data uploaded to the camera
    */
  static void UploadFileToCamera(Arena::IDevice* inDevice,
    CameraFileType inFileType, const void* inData, size_t inDataSize,
    int (*inProgressFunc)(size_t, size_t) = NULL);

  /**
  * @fn void  ReloadNetwork()
  *
  * @param inDevice
  *   - Type: Arena::IDevice*
  *   - Pointer to the IDevice object of ArenaSDK
  * 
  * @return
  *   - none
  *
  * <B> ReloadNetwork </B> instructs the camera to re-uplaod the network
  * into the sensor. This process requires about 30 sec to
  * finish uploading
  */
  static void  ReloadNetwork(Arena::IDevice* inDevice);

// protected:
  // typedefs ------------------------------------------------------------------
  typedef struct
  {
    uint16_t          input_tensor_norm_k[8];         // offset 0   (128) -- k00/k02/k03/k11/k13/k20/k22/k23
  
    uint16_t          dd_ch7_x;                       // offset 16  (144)
    uint16_t          dd_ch7_y;                       // offset 18  (146)
    uint16_t          dd_ch8_x;                       // offset 20  (148)
    uint16_t          dd_ch8_y;                       // offset 22  (150)
  
    uint8_t           input_tensor_format;            // offset 24  (152)
    uint8_t           reserved_0[3];                  // offset 25  (153)
  
    uint16_t          input_tensor_norm_ygain;        // offset 28  (156)
    uint16_t          input_tensor_norm_yadd;         // offset 30  (158)
    uint32_t          y_clip;                         // offset 32  (160)
    uint32_t          cb_clip;                        // offset 36  (164)
    uint32_t          cr_clip;                        // offset 40  (168)
  
    uint16_t          input_norm[4];                  // offset 44  (172)
    uint8_t           input_norm_shift[4];            // offset 52  (180)
    uint32_t          input_norm_clip[4];             // offset 56  (184)
  
    uint32_t          reg_wr_mask;                    // offset 72  (200)
    uint32_t          reserved_1[13];                 // offset 76  (204)
  } fpk_dnn_info;
  
  // fpk_info ver.2.0 (size = 256bytes)
  typedef struct
  {
    // Header part (Version 1 header)
    uint32_t          signature;                      // offset 0
    uint32_t          data_size;                      // offset 4
    uint16_t          version_major;                  // offset 8
    uint16_t          version_minor;                  // offset 10
    uint16_t          chip_id;                        // offset 12
    uint16_t          data_type;                      // offset 14  (section size 16bytes)
  
    //  DNN information
    int8_t            fpk_info_str[64];               // offset  16 (0)
    uint32_t          reserved_v20[11];               // offset  80 (64)
    uint32_t          network_num;                    // offset 124 (108)
  
    //  DNN0 settings
    fpk_dnn_info      dnn[1];                         // offset 128 (0)
  } fpk_info;

  // Member variables ----------------------------------------------------------
  bool  mVerboseMode;
  Arena::IDevice  *mDevice;
  GenApi::INodeMap *mNodeMap;
  fpk_info  mFPKinfo;
  bool mNoRawOutput;

  // Protected static member functions -----------------------------------------
  static void RetrieveFPKinfo(GenApi::INodeMap *inNodeMap, fpk_info *outInfo);
  static uint8_t *RetrieveLabelData(GenApi::INodeMap *inNodeMap, size_t *outDataSize);
  static void MakeLabelList(const uint8_t *inLabelData, size_t inDataSize,
                                  std::vector<std::string> *outList);
  static bool ValidateFPKinfo(const fpk_info *inInfo);
  static void DumpFPKinfo(const fpk_info *inInfo);
  static void ReadFile(GenApi::INodeMap *inNodeMap,
                              const char *inFileName,
                              void *inFileBuf, size_t inFileSize,
                              int inOffset = 0);
  static int64_t GetFileSize(GenApi::INodeMap *inNodeMap,
                              const char *inFileName);
  static void WriteFile(GenApi::INodeMap* inNodeMap,
                              const char* inFileName,
                              const void* inFileBuf, size_t inFileSize,
                              int inOffset = 0,
                              int (*inProgressFunc)(size_t, size_t) = NULL);
  static void SetEnumNode(GenApi::INodeMap *inNodeMap,
                              const char *inNodeName, const char *inValue);
  static void SetFloatNode(GenApi::INodeMap* inNodeMap,
                    const char* inNodeName, double inValue);
  static void SetBooleanNode(GenApi::INodeMap* inNodeMap,
                    const char* inNodeName, bool inValue);
  static void SetIntNode(GenApi::INodeMap *inNodeMap,
                              const char *inNodeName, int64_t inValue);
  static int64_t GetIntNode(GenApi::INodeMap *inNodeMap,
                              const char *inNodeName);
  static void ExecuteNode(GenApi::INodeMap *inNodeMap,
                              const char *inNodeName);
};

// Namespace -------------------------------------------------------------------
}
#endif //ARENA_EXAMPLE_IMX501_UTILS_H
