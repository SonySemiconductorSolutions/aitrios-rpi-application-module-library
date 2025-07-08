---
title: Triton® Smart
sidebar_position: 1
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ApiLink from '@site/src/components/ApiLink';

# Triton® Smart (IMX501)

## Installation

### 1) Hardware setup

We recommend a PoE (IEEE 802.3af) setup where both the Triton® Smart camera and the host device are connected.
```
[PC] --- [POE] --- [Triton® Camera]
```
IPv4 should be set to Manual, Address: 169.254.0.1, Netmask: 255.255.0.0  

System requirements (LUCID Arena compatible device as stated by the [Arena SDK docs](https://support.thinklucid.com/arena-sdk-documentation/)):
- Linux: x64 - Ubuntu 22.04/24.04 (64-bit)
- Linux: arm64 - Ubuntu 22.04/24.04 (64-bit)
- Windows: amd64 - Windows 11 (64-bit) & Windows 10 (32/64-bit)

### 2) Verify your setup

Required Python version: 3.11

If you haven't already done so, make sure to install modlib (in your virtual environment).  
(Optional) Create and enable a virtual environment.  

```shell
python -m venv .venv
source .venv/bin/activate
```

One can use pip to install the library in your project Python environment.
```shell
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git
```

:::warning  
We are currently working on publishing modlib to the PyPI index. This will make it easier for developers to install and manage the library using pip and allow for a more streamlined setup process. Stay tuned for updates!
:::

Let's verify that our camera is connected and the Application Module Library is installed correctly.  
Create a new Python file named `test_connection.py`. and run the following code to see a camera preview.

```python title="test_connection.py"
from modlib.devices import Triton

if __name__ == "__main__":
    device = Triton()
    device.test_connection()
```

Expected output: 
```
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

Incorrect outputoutput:
```
Standard exception thrown: 
terminate called after throwing an instance of 'std::runtime_error'
  what():  deviceInfos.size() == 0, no device connected
Aborted
```
Check the network settings for your network card: IPv4 should be set to Manual, Address: 169.254.0.1, Netmask: 255.255.0.0


## Camera Specifications

| Camera | | |
|----------------------------|--------------------------|-----------|
| Resolution                 |                          | 4056(H)×3040(V), 12.3 MP |
| Sensor size                |                          | Diagonal 7.857 mm (1/2.3 type) |
| Pixel size                 |                          | 1.55 μm (H) × 1.55 μm (V) |
| Frame rate                 | 12.3 MP, 4052x3036       | ~8 fps (Full Resolution) |
| Request rate (rps modlib)  | 12.3 MP, 4052x3036       | 4+ rps (Full Resolution) |
|                            | 3.07 MP, 2024×1516       | 15+ rps (2×2 binned) |
|                            | 1.36 MP, 1348×1008       | 23+ rps (3×3 binned) |
|                            | 0.77 MP, 1012×756        | ~30 rps (4×4 binned) |
|                            | 0.49 MP, 808×604         | 30+ rps (5×5 binned) |
|                            | 0.34 MP, 672×504         | 30+ rps (6×6 binned) |
|                            | 0.25 MP, 576×432         | 30+ rps (7×7 binned) |
|                            | 0.19 MP, 504×378         | 30+ rps (8×8 binned) |
| Power Requirement          |                          | PoE (IEEE 802.3af) or 12-24 VDC external |
| Power Consumption          |                          | 3.1W via PoE, 2.5W when powered externally |

As released by LUCID Vision Labs. [https://thinklucid.com/product/triton-smart-12-3-mp-imx501/](https://thinklucid.com/product/triton-smart-12-3-mp-imx501/)

| IMX500 | | |
|----------------------------|--------------------------|-----------|
| Input tensor size          | Min                      | RGB: 64(H) x 64(V) |
|                            | Max                      | RGB: 640 x 640 |
| Input data type            |                          | int8 or uint8 |
| Supported ML frameworks    |                          | PyTorch (ONNX), TensorFlow (Keras) |
| Memory size                | network + working memory | 8,388,480 bytes (8 MiB)  |
| DSP (*)                    | Number of MACs           | 2304 MACs|
|                            | Peak Power efficiency    | 4.97 TOPS / W |
|                            | Procesor Clock           | 262.5 MHz |
| Sensor Readout time (*)    | 12.3 MP, 4056×3040       | 16.52 msec |
|                            | 3.1 MP, 2028×1520        | 5.52 msec |
| Power Consumption (*)      | 12.3 MP, 4056×3040       | 278.8 mW |
|                            | 3.1 MP, 2028×1520        | 379.1 mW |

(*) As published in: [https://ieeexplore.ieee.org/document/9365965](https://ieeexplore.ieee.org/document/9365965)  
For more information see: [https://developer.sony.com/imx500/imx500-key-specifications](https://developer.sony.com/imx500/imx500-key-specifications)
