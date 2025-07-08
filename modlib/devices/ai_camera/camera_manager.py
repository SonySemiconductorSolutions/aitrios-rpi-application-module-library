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

import os
import selectors
import threading

import numpy as np

from .utils import convert_from_libcamera_type, libcamera


class LibcameraRequest:
    def __init__(self, request, device):
        self.request = request
        self.device = device
        self.ref_count = 1

        with self.device.req_lock:
            self.syncs = [
                self.device.allocator.sync(self.device.allocator, buffer, False)
                for buffer in self.request.buffers.values()
            ]
            self.device.allocator.acquire(self.request.buffers)
            [sync.__enter__() for sync in self.syncs]

        self._metadata = None

    def acquire(self):
        with self.device.req_lock:
            if self.ref_count == 0:
                raise RuntimeError("CompletedRequest: acquiring lock with ref_count 0")
            self.ref_count += 1

    def release(self):
        with self.device.req_lock:
            self.ref_count -= 1
            if self.ref_count == 0:
                # Recycle (if the camera has not been stopped since the request was returned)
                if self.device._running:
                    self.request.reuse()
                    for id, value in self.device.config.libcamera_controls.items():
                        self.request.set_control(id, value)
                    self.device.camera.queue_request(self.request)

                [sync.__exit__() for sync in self.syncs]
                self.device.allocator.release(self.request.buffers)
                self.request = None
                self._metadata = None

            elif self.ref_count < 0:
                raise RuntimeError("CompletedRequest: lock now has negative ref_count")

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {}
            for k, v in self.request.metadata.items():
                self._metadata[k.name] = convert_from_libcamera_type(v)
        return self._metadata

    @property
    def image(self):
        # stream = "main"
        stream = self.device.config.libcamera_config.at(0).stream  # index 0: "main" stream
        _fb = self.request.buffers[stream]
        _sync = self.device.allocator.sync(self.device.allocator, _fb, False)
        b = _sync.__enter__()

        array = np.array(b, copy=True, dtype=np.uint8)

        cfg_main = self.device.config.camera_config["main"]
        fmt, (w, h), stride = cfg_main["format"], cfg_main["size"], cfg_main["stride"]

        if fmt in ("BGR888", "RGB888"):
            if stride != w * 3:
                array = array.reshape((h, stride))
                array = array[:, : w * 3]
            array = array.reshape((h, w, 3))
        else:
            raise RuntimeError("Format " + fmt + " not supported")

        _sync.__exit__()
        return array


class CameraManager:
    def __init__(self):
        self.cameras = {}
        self._lock = threading.Lock()
        self._running = False
        self._cms = None

    def add(self, camera):
        if not self.is_valid(camera):
            raise ValueError("Camera object not valid.")
        index = camera.camera_num

        with self._lock:
            self.cameras[index] = camera
            if not self._running:
                self.thread = threading.Thread(target=self.listen, daemon=True)
                self._running = True
                self.thread.start()

    @staticmethod
    def is_valid(camera):
        if not hasattr(camera, "camera_num") or camera.camera_num is None:
            return False
        if not hasattr(camera, "_requests"):
            return False
        if not hasattr(camera, "_requestslock"):
            return False
        if not hasattr(camera, "notifyme_w"):
            return False
        if not hasattr(camera, "req_lock"):
            return False
        return True

    @property
    def cms(self):
        if self._cms is None:
            self._cms = libcamera.CameraManager.singleton()
        return self._cms

    @property
    def global_cameras(self):
        def describe_camera(cam, num):
            info = {k.name: v for k, v in cam.properties.items() if k.name in ("Model", "Location", "Rotation")}
            info["Id"], info["Num"] = cam.id, num
            return info

        # Sort alphabetically so they are deterministic, but send USB cams to the back of the class.
        cameras = [describe_camera(cam, i) for i, cam in enumerate(self.cms.cameras)]
        return sorted(cameras, key=lambda cam: ("/usb" not in cam["Id"], cam["Id"]), reverse=True)

    def cleanup(self, index):
        flag = False
        with self._lock:
            del self.cameras[index]
            if self.cameras == {}:
                self._running = False
                flag = True
        # Don't join inside _lock to prevent potential deadlock on close
        if flag:
            self.thread.join()
            self._cms = None

    def listen(self):
        # Libcamera event listener
        sel = selectors.DefaultSelector()
        sel.register(self.cms.event_fd, selectors.EVENT_READ, self.handle_request)

        while self._running:
            events = sel.select(0.2)
            for key, _ in events:
                callback = key.data
                callback()

        sel.unregister(self.cms.event_fd)
        self._cms = None

    def handle_request(self, flushid=None):
        # Add requests to the cameras request buffer
        with self._lock:
            cams = set()
            for req in self.cms.get_ready_requests():
                if req.status == libcamera.Request.Status.Complete and req.cookie != flushid:
                    cams.add(req.cookie)
                    with self.cameras[req.cookie]._requestslock:
                        self.cameras[req.cookie]._requests += [LibcameraRequest(req, self.cameras[req.cookie])]
                        self.cameras[req.cookie].rps.update()
            # Notify a request is ready to be processed to the designated camera
            for c in cams:
                os.write(self.cameras[c].notifyme_w, b"\x00")
