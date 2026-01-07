<h1 align='center'><b>Application Module Library</b></h1>
<p align='center'>
Application Module Library (modlib) is an SDK designed to simplify and streamline the process of creating <b>end-to-end</b> applications for the <b>IMX500 vision sensor</b>.
</p>

## 1. Quickstart

For the Raspberry Pi AI Camera, make sure that the python version of the virtual environment is the same as the system Python version. (Verify by running `/usr/bin/python3 -V`). This depends on your platform.
- Raspberry Pi OS Bookworm: **Python 3.11**
- Raspberry Pi OS Trixie: **Python 3.13**  

We expect the Raspberry Pi computer to be installed correclty with the AiCamera connected and access to the board terminal. Ensure that your Raspberry Pi runs the latest software. A full setup guide can be found [here](docs/src/devices/ai_camera.md).

One can use pip to install the library in your project Python environment.
```shell
pip install modlib
```

Create a new Python file named `hello_world.py`. and run the following code to see a camera preview.
```python title="hello_world.py"
from modlib.devices import AiCamera

device = AiCamera()

with device as stream:
    for frame in stream:
        frame.display()
```

## 2. Documentation

For a comprehensive guide on how to use the library, including detailed API documentation, tutorials, and examples, make sure to check out the documentation!

1. **Documentation site**: Coming soon!
2. **Local Documentation**: You can run a local documentation server using:
```
python -m http.server --directory ./docs
```

## 3. Development Environment Setup (Build from source)

Before getting started, ensure you have the following prerequisites installed on your system:

### 3.1 Prerequisites

#### 3.1.1 **UV**

UV is the Python environment and package manager used for the project. Follow the installation instructions provided in the [UV documentation](https://docs.astral.sh/uv/getting-started/installation/) to set it up.
Verify your uv installation by running:
```
uv --version
```

#### 3.1.2 **Build environment dependencies**
- On Linux, install ninja using apt. And make sure to have a gcc compiler installed on your system: `sudo apt install ninja-build`
- On Windows, install ninja using Chocolatey. And make sure to have an MSVC compiler installed on your system: `choco install ninja`

### 3.2 Hardware setup

#### 3.2.1 Raspberry Pi AI Camera
We expect the Raspberry Pi computer to be installed correclty with the AiCamera connected and access to the board terminal. Ensure that your Raspberry Pi runs the latest software:
```
sudo apt update && sudo apt full-upgrade
sudo apt install imx500-all
```
Reboot if needed.

#### 3.2.2 Triton® Smart (IMX501)

> NOTE: It is possible to skip this step if you do not intend to use the Triton® Smart device. Be aware that any attempt to initialize the Triton® device will fail if you chose to skip the installation of the Arena SDK.

We recommend a PoE (IEEE 802.3af) setup where both the Triton® Smart camera and the host device are connected.
`[PC] --- [POE] --- [Triton® Camera]`
IPv4 should be set to Manual, Address: 169.254.0.1, Netmask: 255.255.0.0  

System requirements (LUCID Arena compatible device as stated by the [Arena SDK docs](https://support.thinklucid.com/arena-sdk-documentation/)):
- Linux: x64 - Ubuntu 22.04/24.04 (64-bit)
- Linux: arm64 - Ubuntu 22.04/24.04 (64-bit)
- Windows: amd64 - Windows 11 (64-bit) & Windows 10 (32/64-bit)

**Install the Arena SDK** (Linux x64 or ARM64 or Windows)
Download from: https://thinklucid.com/downloads-hub/  

- **Windows**: Run the ArenaSDK installer for profile: `Developer`.  
- **Linux**: Unzip and make sure to make the libraries (.so) available.
Example Linux x64 (but similar for aarch64):
```shell
tar -xvzf ArenaSDK_<sdk-version-number>_Linux_x64.tar.gz
cd /path/to/ArenaSDK_Linux
sudo sh Arena_SDK_Linux_x64.conf
```

### 3.3 Setup

```
uv sync
```

This will setup and build the project using the meson-python build system.


### 3.4 Running examples

As a basic example let's demonstrate the usage of the Raspberry Pi AiCamera device using a pre-trained SSDMobileNetV2FPNLite320x320 object detection model.

1. Run any of the model examples from `./examples/aicam`:

```
uv run examples/aicam/classifier.py
```
```
uv run examples/aicam/detector.py
```
```
uv run examples/aicam/segment.py
```
```
uv run examples/aicam/posenet.py
```

Note that the Application Module Library API allows you to create custom Models and combine any network.rpk with your own custom post_processing function. More information in `Docs > getting_started > custom_models.md`.

2. Run any of the application examples `./examples/apps`:

```
uv run examples/apps/tracker.py
```
```
uv run examples/apps/area.py
```
```
uv run examples/apps/heatmap.py
```

...

## Extra
- Unit-tests are included in the corresponding `/tests` folder and can be exectued by running: `make test`  
- Linting corrections and checks using ruff by running: `make lint`
- Building the Python wheel.
One can create a wheel file of the Application Module Library by running: `make build`
The generated wheel file located in the `/dist` folder can be used to install the library in you project environment.


## Releases

Release tags must be of the format "\d+\.\d+\.\d+" example "1.0.4".

## License

[LICENSE](./LICENSE)

## Trademarks

[Trademarks | Sony Semiconductor Solutions Group](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/trademarks)

## Notice

Sony Semiconductor Solutions Corporation assumes no responsibility for applications created using this library. Use of the library is entirely at the user's own risk.

### Security

Please read the Site Policy of GitHub and understand the usage conditions.
