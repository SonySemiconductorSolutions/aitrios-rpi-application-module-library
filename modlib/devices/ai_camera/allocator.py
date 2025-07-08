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
import mmap
import os

from .v4l2 import _IOW, _IOWR
from .utils import libcamera

try:
    import fcntl
except ImportError:
    fcntl = None

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s]  %(levelname)s %(message)s")
logger = logging.getLogger(__name__.split(".")[-1])


heapNames = ["/dev/dma_heap/vidbuf_cached", "/dev/dma_heap/linux,cma"]


# Kernel stuff from linux/dma-buf.h
class dma_buf_sync(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
    ]


DMA_BUF_SYNC_READ = 1 << 0
DMA_BUF_SYNC_WRITE = 2 << 0
DMA_BUF_SYNC_RW = DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE
DMA_BUF_SYNC_START = 0 << 2
DMA_BUF_SYNC_END = 1 << 2

DMA_BUF_BASE = "b"
DMA_BUF_IOCTL_SYNC = _IOW(DMA_BUF_BASE, 0, dma_buf_sync)

DMA_BUF_SET_NAME = _IOW(DMA_BUF_BASE, 1, ctypes.c_char_p)


# Kernel stuff from linux/dma-heap.h
class dma_heap_allocation_data(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_uint64),
        ("fd", ctypes.c_uint32),
        ("fd_flags", ctypes.c_uint32),
        ("heap_flags", ctypes.c_uint64),
    ]


DMA_HEAP_IOC_MAGIC = "H"

DMA_HEAP_IOCTL_ALLOC = _IOWR(DMA_HEAP_IOC_MAGIC, 0, dma_heap_allocation_data)


# Libcamera C++ classes
class UniqueFD:
    """Libcamera UniqueFD Class"""

    def __init__(self, fd=-1):
        if isinstance(fd, UniqueFD):
            self.__fd = fd.release()
        else:
            self.__fd = fd

    def release(self):
        fd = self.__fd
        self.__fd = -1
        return fd

    def get(self):
        return self.__fd

    def isValid(self):
        return self.__fd >= 0


class DmaHeap:
    """DmaHeap"""

    def __init__(self):
        self.__dmaHeapHandle = UniqueFD()
        for name in heapNames:
            try:
                ret = os.open(name, os.O_CLOEXEC | os.O_RDWR)
            except FileNotFoundError:
                logger.error(f"Failed to open {name}")
                continue

            self.__dmaHeapHandle = UniqueFD(ret)
            break

        if not self.__dmaHeapHandle.isValid():
            raise RuntimeError("Could not open any dmaHeap device")

    @property
    def isValid(self):
        return self.__dmaHeapHandle.isValid()

    def alloc(self, name, size) -> UniqueFD:
        alloc = dma_heap_allocation_data()
        alloc.len = size
        alloc.fd_flags = os.O_CLOEXEC | os.O_RDWR

        ret = fcntl.ioctl(self.__dmaHeapHandle.get(), DMA_HEAP_IOCTL_ALLOC, alloc)
        if ret < 0:
            logger.error(f"dmaHeap allocation failure for {name}")
            return UniqueFD()

        allocFd = UniqueFD(alloc.fd)
        ret = fcntl.ioctl(allocFd.get(), DMA_BUF_SET_NAME, name)
        if not isinstance(ret, bytes) and ret < 0:
            logger.error(f"dmaHeap naming failure for {name}")
            return UniqueFD()

        return allocFd

    def close(self):
        os.close(self.__dmaHeapHandle.get())


# BASE ALLOCATOR


class Allocator:
    """Base class for allocators"""

    def __init__(self):
        self.sync = Sync

    def allocate(self, libcamera_config, use_case):
        pass

    def buffers(self, stream):
        pass

    def acquire(self, bufs):
        pass

    def release(self, bufs):
        pass

    def close(self):
        pass


class Sync:
    """Base class for allocator syncronisations"""

    def __init__(self, allocator, fb, write):
        self.__fb = fb

    def __enter__(self):
        import mmap

        # Check if the buffer is contiguous and find the total length.
        fd = self.__fb.planes[0].fd
        planes_metadata = self.__fb.metadata.planes
        buflen = 0
        for p, p_metadata in zip(self.__fb.planes, planes_metadata):
            # bytes_used is the same as p.length for regular frames, but correctly reflects
            # the compressed image size for MJPEG cameras.
            buflen = buflen + p_metadata.bytes_used
            if fd != p.fd:
                raise RuntimeError("_MappedBuffer: Cannot map non-contiguous buffer!")

        self.__mm = mmap.mmap(fd, buflen, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        return self.__mm

    def __exit__(self, exc_type=None, exc_value=None, exc_traceback=None):
        if self.__mm is not None:
            self.__mm.close()


# DMA ALLOCATOR


class DmaAllocator(Allocator):
    """DmaHeap Allocator"""

    def __init__(self):
        super().__init__()
        self.dmaHeap = DmaHeap()
        self.mapped_buffers = {}
        self.mapped_buffers_used = {}
        self.frame_buffers = {}
        self.open_fds = []
        self.libcamera_fds = []
        self.sync = self.DmaSync

    def allocate(self, libcamera_config, _):
        # Delete old buffers
        self.libcamera_fds = []
        self.cleanup()
        # Close our copies of fds
        for fd in self.open_fds:
            os.close(fd)
        self.frame_buffers = {}
        self.open_fds = []

        for c, stream_config in enumerate(libcamera_config):
            stream = stream_config.stream
            fb = []
            for i in range(stream_config.buffer_count):
                fd = self.dmaHeap.alloc(f"modlib-{i}", stream_config.frame_size)
                # Keep track of our allocated fds, as libcamera makes copies
                self.open_fds.append(fd.get())

                if not fd.isValid():
                    raise RuntimeError(f"failed to allocate capture buffers for stream {c}")

                plane = [libcamera.FrameBuffer.Plane()]
                plane[0].fd = fd.get()
                plane[0].offset = 0
                plane[0].length = stream_config.frame_size

                self.libcamera_fds.append(plane[0].fd)
                self.mapped_buffers_used[plane[0].fd] = False

                fb.append(libcamera.FrameBuffer(plane))
                memory = mmap.mmap(
                    plane[0].fd, stream_config.frame_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
                )
                self.mapped_buffers[fb[-1]] = memory

            self.frame_buffers[stream] = fb
            msg = (
                f"Allocated {len(fb)} buffers for stream {c} with fds "
                f"{[f.planes[0].fd for f in self.frame_buffers[stream]]}"
            )
            logger.debug(msg)

    def buffers(self, stream):
        return self.frame_buffers[stream]

    def acquire(self, buffers):
        for buffer in buffers.values():
            fd = buffer.planes[0].fd
            self.mapped_buffers_used[fd] = True

    def release(self, buffers):
        for buffer in buffers.values():
            fd = buffer.planes[0].fd
            self.mapped_buffers_used[fd] = False
        self.cleanup()

    def cleanup(self):
        for k, v in self.mapped_buffers.items():
            fd = k.planes[0].fd
            if not self.mapped_buffers_used[fd] and fd not in self.libcamera_fds:
                # Not in use by any requests, and not currently allocated
                v.close()
                del self.mapped_buffers_used[fd]
        for k in [k for k, v in self.mapped_buffers.items() if v.closed]:
            del self.mapped_buffers[k]

    def close(self):
        self.libcamera_fds = []
        self.cleanup()
        # Close our copies of fds
        for fd in self.open_fds:
            os.close(fd)
        self.frame_buffers = {}
        self.open_fds = []
        if self.dmaHeap is not None:
            self.dmaHeap.close()

    def __del__(self):
        self.close()

    class DmaSync(Sync):
        """Dma Buffer Sync"""

        def __init__(self, allocator, fb, write):
            self.allocator = allocator
            self.__fb = fb
            self.__write = write

        def __enter__(self):
            dma_sync = dma_buf_sync()
            dma_sync.flags = DMA_BUF_SYNC_START | (DMA_BUF_SYNC_RW if self.__write else DMA_BUF_SYNC_READ)

            it = self.allocator.mapped_buffers.get(self.__fb, None)
            if it is None:
                raise RuntimeError("failed to find buffer in DmaSync")

            ret = fcntl.ioctl(self.__fb.planes[0].fd, DMA_BUF_IOCTL_SYNC, dma_sync)
            if ret:
                raise RuntimeError("failed to lock-sync-write dma buf")
            return it

        def __exit__(self, exc_type=None, exc_value=None, exc_traceback=None):
            dma_sync = dma_buf_sync()
            dma_sync.flags = DMA_BUF_SYNC_END | (DMA_BUF_SYNC_RW if self.__write else DMA_BUF_SYNC_READ)

            ret = fcntl.ioctl(self.__fb.planes[0].fd, DMA_BUF_IOCTL_SYNC, dma_sync)
            if ret:
                logging.error("failed to unlock-sync-write dma buf")
