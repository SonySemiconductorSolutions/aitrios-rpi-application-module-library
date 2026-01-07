---
title: AiCamera
sidebar_position: 0
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import ApiLink from '@site/src/components/ApiLink';

# Raspberry Pi AI Camera

Compatible with all Raspberry Pi computers as host, the [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/) takes advantage of Sony’s IMX500 Intelligent Vision Sensor to help you create impressive vision AI applications and neural network models using the on-module AI processor.

## Installation

### 1) Hardware setup

We expect the Raspberry Pi computer to be installed correclty with the AiCamera connected and access to the board terminal. For a detailed description on how to install a Raspberry Pi Camera see [this](https://www.raspberrypi.com/documentation/accessories/camera.html#install-a-raspberry-pi-camera) link.

### 2) Install IMX500 Firmware

Ensure that your Raspberry Pi runs the latest software:

```shell
sudo apt update && sudo apt full-upgrade
sudo apt install imx500-all
```

**This command:**
- installs the /lib/firmware/imx500_loader.fpk and /lib/firmware/imx500_firmware.fpk firmware files required to operate the IMX500 sensor
- places a number of neural network model firmware files in /usr/share/imx500-models/
- installs the Sony network model packaging tools

Now that you’ve installed the prerequisites, restart your Raspberry Pi:
```shell
sudo reboot
```

### 3) Verify your setup

If you haven't already done so, make sure to install modlib (in your virtual environment).
The RPi AI Camera requires the **system Python version** to interface with libcamera. (Verify by running `/usr/bin/python3 -V`). This depends on your platform.
- Raspberry Pi OS Bookworm: **Python 3.11**
- Raspberry Pi OS Trixie: **Python 3.13**  

```shell
python -m venv .venv
source .venv/bin/activate
```
Make sure that the python version of the virtual environment is the same as the system Python version. (`/usr/bin/python3 -V`)  

One can use pip to install the library in your project Python environment.
```shell
pip install modlib
```

Let's verify that our camera is connected and the Application Module Library is installed correctly.  
Create a new Python file named `hello_world.py`. and run the following code to see a camera preview.

```python title="hello_world.py"
from modlib.devices import AiCamera

device = AiCamera()

with device as stream:
    for frame in stream:
        frame.display()
```


## Camera Specifications

| Camera | | |
|----------------------------|--------------------------|-----------|
| Resolution                 |                          | 4056(H)×3040(V), 12.3 MP |
| Sensor size                |                          | Diagonal 7.857 mm (1/2.3 type) |
| Pixel size                 |                          | 1.55 μm (H) × 1.55 μm (V) |
| Field of View              |                          | 78.3 (±3) degree FoV with manual/mechanical adjustable focus |
| F-stop (Aperture)          |                          | F1.79 Focal Ratio |
| Frame rate                 | 12.3 MP, 4056×3040       | 10 fps (Full Resolution) |
|                            | 3.1 MP, 2028×1520        | 30 fps (2×2 binned) |
| Request rate (rps modlib)  | 12.3 MP, 4056×3040       | 10 rps (Full Resolution) |
|                            | 3.1 MP, 2028×1520        | 30 rps (2×2 binned) |

As published in: [https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf)

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