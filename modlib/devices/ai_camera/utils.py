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

import importlib.util
import os
import sys
import sysconfig


class Fallback:
    def __init__(self, pkg, pkg_path) -> None:
        self.pkg = pkg
        self.pkg_path = pkg_path

    def __getattr__(self, name):
        raise ImportError(
            f"{self.pkg} module is not available. Check if it is installed and the path is correct: {self.pkg_path}"
        )


class VersionMismatchFallback:
    def __init__(self, pkg, pkg_path) -> None:
        self.pkg = pkg
        self.pkg_path = pkg_path
        self.current_version = sys.version_info[:2]

    def __getattr__(self, name):
        current_ver_str = f"{self.current_version[0]}.{self.current_version[1]}"
        available_files = [f for f in os.listdir(self.pkg_path) if f.startswith("_libcamera.") and f.endswith(".so")]
        files_list = "\n  ".join(sorted(available_files)) if available_files else "  (none found)"

        raise ImportError(
            f"\n{self.pkg} module encountered mismatching Python versions. "
            f"The module was compiled for a different Python version than the current interpreter "
            f"(Python {current_ver_str}).\n\n"
            f"Available libcamera packages in {self.pkg_path}:\n  {files_list}\n\n"
            f"Please switch to a Python interpreter version that matches one of the available modules."
        )


def _abi_mismatch(package_path) -> bool:
    # Current SOABI of the python interpreter
    current_soabi = sysconfig.get_config_var("SOABI")

    # Determine SOABI for all _libcamera.*.so packages in the package directory
    package_soabis = []
    for filename in os.listdir(package_path):
        if filename.startswith("_libcamera.") and filename.endswith(".so"):
            # Extract SOABI from filename: _libcamera.{SOABI}.so
            package_soabis.append(filename[len("_libcamera.") : -len(".so")])

    abi_available = current_soabi in package_soabis
    return not abi_available


def _import_global_package(package_name, package_path):
    if not os.path.exists(package_path):
        return Fallback(package_name, package_path)

    spec = importlib.util.spec_from_file_location(package_name, f"{package_path}/__init__.py")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = pkg
    try:
        spec.loader.exec_module(pkg)
    except Exception:
        if _abi_mismatch(package_path):
            return VersionMismatchFallback(package_name, package_path)
        else:
            return Fallback(package_name, package_path)  # regular error Fallback

    return pkg


def _import_libcamera():
    # Import from globally installed package
    libcamera = _import_global_package("libcamera", "/usr/lib/python3/dist-packages/libcamera")
    if isinstance(libcamera, (Fallback, VersionMismatchFallback)):
        return libcamera

    # LIBCAMERA CONFIGURATION
    def libcamera_transforms_eq(t1, t2):
        return t1.hflip == t2.hflip and t1.vflip == t2.vflip and t1.transpose == t2.transpose

    def libcamera_colour_spaces_eq(c1, c2):
        return (
            c1.primaries == c2.primaries
            and c1.transferFunction == c2.transferFunction
            and c1.ycbcrEncoding == c2.ycbcrEncoding
            and c1.range == c2.range
        )

    def _libcamera_size_to_tuple(sz):
        return (sz.width, sz.height)

    def _libcamera_rect_to_tuple(rect):
        return (rect.x, rect.y, rect.width, rect.height)

    libcamera.Transform.__repr__ = libcamera.Transform.__str__
    libcamera.Transform.__eq__ = libcamera_transforms_eq
    libcamera.ColorSpace.__repr__ = libcamera.ColorSpace.__str__
    libcamera.ColorSpace.__eq__ = libcamera_colour_spaces_eq
    libcamera.Size.to_tuple = _libcamera_size_to_tuple
    libcamera.Rectangle.to_tuple = _libcamera_rect_to_tuple

    return libcamera


libcamera = _import_libcamera()


# LIBCAMERA CONFIGURATION UTILS
def transform_to_orientation(transform):
    _TRANSFORM_TO_ORIENTATION_TABLE = {
        libcamera.Transform(): libcamera.Orientation.Rotate0,
        libcamera.Transform(hflip=1): libcamera.Orientation.Rotate0Mirror,
        libcamera.Transform(vflip=1): libcamera.Orientation.Rotate180Mirror,
        libcamera.Transform(hflip=1, vflip=1): libcamera.Orientation.Rotate180,
        libcamera.Transform(transpose=1): libcamera.Orientation.Rotate90Mirror,
        libcamera.Transform(transpose=1, hflip=1): libcamera.Orientation.Rotate270,
        libcamera.Transform(transpose=1, vflip=1): libcamera.Orientation.Rotate90,
        libcamera.Transform(transpose=1, hflip=1, vflip=1): libcamera.Orientation.Rotate270Mirror,
    }

    # A transform is an object and not a proper dictionary key, so must search by hand.
    if isinstance(transform, libcamera.Transform):
        for k, v in _TRANSFORM_TO_ORIENTATION_TABLE.items():
            if k == transform:
                return v
    raise RuntimeError(f"Unknown transform {transform}")


def orientation_to_transform(orientation):
    _ORIENTATION_TO_TRANSFORM_TABLE = {
        libcamera.Orientation.Rotate0: libcamera.Transform(),
        libcamera.Orientation.Rotate0Mirror: libcamera.Transform(hflip=1),
        libcamera.Orientation.Rotate180Mirror: libcamera.Transform(vflip=1),
        libcamera.Orientation.Rotate180: libcamera.Transform(hflip=1, vflip=1),
        libcamera.Orientation.Rotate90Mirror: libcamera.Transform(transpose=1),
        libcamera.Orientation.Rotate270: libcamera.Transform(transpose=1, hflip=1),
        libcamera.Orientation.Rotate90: libcamera.Transform(transpose=1, vflip=1),
        libcamera.Orientation.Rotate270Mirror: libcamera.Transform(transpose=1, hflip=1, vflip=1),
    }

    # Return a copy of the object.
    return libcamera.Transform(_ORIENTATION_TO_TRANSFORM_TABLE[orientation])


def convert_from_libcamera_type(value):
    if isinstance(value, libcamera.Rectangle):
        value = value.to_tuple()
    elif isinstance(value, libcamera.Size):
        value = value.to_tuple()
    elif isinstance(value, (list, tuple)) and all(isinstance(item, libcamera.Rectangle) for item in value):
        value = [v.to_tuple() for v in value]
    return value


def colour_space_to_libcamera(colour_space, format):
    RGB_FORMATS = {"BGR888", "RGB888", "XBGR8888", "XRGB8888", "RGB161616", "BGR161616"}

    def is_RGB(fmt: str) -> bool:
        return fmt in RGB_FORMATS

    # libcamera may complain if we supply an RGB format stream with a YCbCr matrix or range.
    if is_RGB(format):
        colour_space = libcamera.ColorSpace(colour_space)  # it could be shared with other streams, so copy it
        colour_space.ycbcrEncoding = libcamera.ColorSpace.YcbcrEncoding.Null
        colour_space.range = libcamera.ColorSpace.Range.Full
    return colour_space


def colour_space_from_libcamera(colour_space):
    COLOUR_SPACE_TABLE = {libcamera.ColorSpace.Sycc(), libcamera.ColorSpace.Smpte170m(), libcamera.ColorSpace.Rec709()}

    # Colour spaces may come back from libcamera without a YCbCr matrix or range, meaning
    # they don't look like the 3 standard colour spaces (in the table) that we expect people
    # to use. Let's fix that.
    if colour_space is None:  # USB webcams might have a "None" colour space
        return None
    for cs in COLOUR_SPACE_TABLE:
        if colour_space.primaries == cs.primaries and colour_space.transferFunction == cs.transferFunction:
            return cs
    return colour_space
