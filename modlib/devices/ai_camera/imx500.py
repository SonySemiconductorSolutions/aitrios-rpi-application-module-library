#
# BSD 2-Clause License
#
# Copyright (c) 2021, Raspberry Pi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import ctypes
import logging
import multiprocessing
import os
import sys
import time

from tqdm import tqdm

from .v4l2 import (
    VIDIOC_S_CTRL,
    VIDIOC_S_EXT_CTRLS,
    VIDIOC_G_EXT_CTRLS,
    v4l2_control,
    v4l2_ext_control,
    v4l2_ext_controls,
)
from .utils import libcamera

try:
    import fcntl
except ImportError:
    fcntl = None

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s]  %(levelname)s %(message)s")
logger = logging.getLogger(__name__.split(".")[-1])


SENSOR_W = 4056
SENSOR_H = 3040

FW_NETWORK_STAGE = 2
GET_DEVICE_ID_CTRL_ID = 0x00982902
NETWORK_FW_FD_CTRL_ID = 0x00982901
ROI_CTRL_ID = 0x00982900


class IMX500:
    def __init__(self, network_file: str, camera_id: str = ""):
        self.device_fd = None
        imx500_device_id = None
        spi_device_id = None

        # Initiate
        for i in range(32):
            test_dir = f"/sys/class/video4linux/v4l-subdev{i}/device"
            module_dir = f"{test_dir}/driver/module"
            id_dir = f"{test_dir}/of_node"

            if (
                os.path.exists(module_dir)
                and os.path.islink(module_dir)
                and os.path.islink(id_dir)
                and "imx500" in os.readlink(module_dir)
            ):
                if camera_id == "" or camera_id in os.readlink(id_dir):
                    self.device_fd = open(f"/dev/v4l-subdev{i}", "rb+", buffering=0)
                    imx500_device_id = os.readlink(test_dir).split("/")[-1]
                    spi_device_id = imx500_device_id.replace("001a", "0040")

                    break

        # Check and finish initialisation
        if self.device_fd is None:
            raise RuntimeError("Could not find requested camera dev-node: imx500. AI Camera not found.")
        if imx500_device_id:
            self.fw_progress = open(f"/sys/kernel/debug/imx500-fw:{imx500_device_id}/fw_progress", "r")
        if spi_device_id:
            self.fw_progress_chunk = open(f"/sys/kernel/debug/rp2040-spi:{spi_device_id}/transfer_progress", "r")

        # Upload network firmware
        self.__set_network_firmware(os.path.abspath(network_file))
        self.p = None

    def __del__(self):
        if self.device_fd:
            self.device_fd.close()

    def get_device_id(self) -> str | None:
        """
        Retrieve the unique IMX500 device ID.

        Returns:
            The device ID as an ASCII string if successful, otherwise None.
        """
        ret = None
        imx500_device_id = ""

        r = (ctypes.c_uint32 * 4)()
        r[0] = 0x0
        r[1] = 0x0
        r[2] = 0x0
        r[3] = 0x0

        c = (v4l2_ext_control * 1)()
        c[0].p_u32 = r
        c[0].id = GET_DEVICE_ID_CTRL_ID
        c[0].size = 16

        ctrl = v4l2_ext_controls()
        ctrl.count = 1
        ctrl.controls = c

        try:
            fcntl.ioctl(self.device_fd, VIDIOC_G_EXT_CTRLS, ctrl)
            for i in range(4):
                ret = ctrl.controls[0].p_u32[i]
                imx500_device_id += "%08X" % ret
        except OSError as err:
            logger.error(f"IMX500: Unable to get device ID from device driver: {err}")
            imx500_device_id = None

        return imx500_device_id

    def get_fw_upload_progress(self, stage_req) -> tuple:
        """Returns the current progress of the fw upload in the form of (current, total)."""
        progress_block = 0
        progress_chunk = 0
        size = 0
        stage = 0

        if self.fw_progress:
            self.fw_progress.seek(0)
            progress = self.fw_progress.readline().strip().split()
            stage = int(progress[0])
            progress_block = int(progress[1])
            size = int(progress[2])

        if self.fw_progress_chunk:
            self.fw_progress_chunk.seek(0)
            progress_chunk = int(self.fw_progress_chunk.readline().strip())

        if stage == stage_req:
            return (min(progress_block + progress_chunk, size), size)
        else:
            return (0, 0)

    def show_network_fw_progress_bar(self):
        self.p = multiprocessing.Process(
            target=self.__do_progress_bar, args=(FW_NETWORK_STAGE, "Network Firmware Upload")
        )
        self.p.start()
        self.p.join(0)

    def __do_progress_bar(self, stage_req, title):
        with tqdm(unit="bytes", unit_scale=True, unit_divisor=1024, desc=title, leave=True) as t:
            last_update = 0
            while True:
                current, total = self.get_fw_upload_progress(stage_req)
                if total:
                    t.total = total
                    t.update(current - last_update)
                    last_update = current
                    if current > 0.95 * total:
                        t.update(total - last_update)
                        break
                time.sleep(0.5)

    def stop_network_fw_progress_bar(self):
        if self.p and self.p.is_alive():
            self.p.terminate()
            self.p.join()
            self.p = None

    def set_inference_roi_abs(self, roi: tuple):
        """
        Set the absolute inference image crop.

        Specify an absolute region of interest in the form a (left, top, width, height) crop for the input inference
        image. The co-ordinates are based on the full sensor resolution.
        """
        roi = libcamera.Rectangle(*roi)
        roi = roi.bounded_to(libcamera.Rectangle(0, 0, SENSOR_W, SENSOR_H))

        r = (ctypes.c_uint32 * 4)()
        r[0] = roi.x
        r[1] = roi.y
        r[2] = roi.width
        r[3] = roi.height

        c = (v4l2_ext_control * 1)()
        c[0].p_u32 = r
        c[0].id = ROI_CTRL_ID
        c[0].size = 16

        ctrl = v4l2_ext_controls()
        ctrl.count = 1
        ctrl.controls = c

        try:
            fcntl.ioctl(self.device_fd, VIDIOC_S_EXT_CTRLS, ctrl)
        except OSError as err:
            logger.error(f"IMX500: Unable to set ROI control in the device driver: {err}")

    def __set_network_firmware(self, network_filename: str):
        """Provides a firmware rpk file to upload to the IMX500."""
        if not os.path.isfile(network_filename):
            raise RuntimeError(f"Firmware file {network_filename} does not exist.")

        fd = os.open(network_filename, os.O_RDONLY)
        if fd:
            ctrl = v4l2_control()
            ctrl.id = NETWORK_FW_FD_CTRL_ID
            ctrl.value = fd

            try:
                fcntl.ioctl(self.device_fd, VIDIOC_S_CTRL, ctrl)
                print(
                    "\n------------------------------------------------------------------------------------------------------------------\n"  # noqa
                    "NOTE: Loading network firmware onto the IMX500 can take several minutes, please do not close down the application."  # noqa
                    "\n------------------------------------------------------------------------------------------------------------------\n",  # noqa
                    file=sys.stderr,
                )  # noqa
            except OSError as err:
                raise RuntimeError(f"IMX500: Unable to set network firmware {network_filename}: {err}")
            finally:
                os.close(fd)
