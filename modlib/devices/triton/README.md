**SETUP**

1. POE + camera + wired network cable + network adapter compatible with device requirements [ArenaSDK docs](https://support.thinklucid.com/arena-sdk-documentation/)
2. triton_firmware_1.1.7
3. IPv4 Method Manual: 
- Address: 169.254.0.1
- Netmask: 255.255.0.0


**Prerequisites**

1. Arena SDK (e.g. Linux x64 or ARM64 or Windows)

Observe: From Spring 2025, Arena SDK does not support Ubuntu20 (glibc >= 2.34).

Download from: https://thinklucid.com/downloads-hub/
Unzip and make sure to make the .dll and .so available

```
tar -xvzf ArenaSDK_<sdk-version-number>_Linux_x64.tar.gz
cd /path/to/ArenaSDK_Linux
sudo sh Arena_SDK_Linux_x64.conf
```

Ensure proper communication with the device:

```
python examples/triton/_test.py
Found number of devices: 1
Available devices:
Device 0:
  VendorName: Lucid Vision Labs
  Model: TRI123S-C
  Serial: 242900808
  IP: 169.254.3.2
  SubnetMask: 255.255.0.0
  DefaultGateway: 0.0.0.0
  MacAddress: 1c:0f:af:ec:75:2c
Automatically selecting 1st device: TRI123S-C, 242900808, 169.254.3.2.

Image acquired, size: (4052, 3036)

Simple acquisition completed
INFO:triton:Triton closed successfully.
```

Incorrect output
```
Standard exception thrown: 
terminate called after throwing an instance of 'std::runtime_error'
  what():  deviceInfos.size() == 0, no device connected
Terminated
```
Check the network settings for your network card: IPv4 should be set to Manual, Address: 169.254.0.1, Netmask: 255.255.0.0, [ArenaSDK docs](https://support.thinklucid.com/arena-sdk-documentation/)

2. OpenSSL development package  

Linux: `sudo apt install libssl-dev`  
Windows: `choco install openssl`

**Limitations**
 * Connection is supported only to 1 Triton® device

**Known errors**

***GenericException (GC_ERR_ERROR): Unable to write port***

This error can appear after deployment of a new model to the Triton® device. This error message can appear if the deployed model was converted for a different Triton® camera, make sure that device id is correct in the lucid converter tool. 
```
GenICam exception thrown: 
terminate called after throwing an instance of 'std::runtime_error'
  what():  GenericException (GC_ERR_ERROR): Unable to write port: No Error (Port.cpp, Write; GenTL::GCWritePort); Error Stack:
  -1001 GenTL::GC_ERR_ERROR(-1001) :
      GenTL::HALGev::WriteReg Ack is invalid: -1019 : address: 0x12900004, ACK = 0x83, status = 0x8fff, ack_id = 0x1fd4, length = 0x4
 (file 'Port.cpp', line 79)
Terminated
```
