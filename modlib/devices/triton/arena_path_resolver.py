# -----------------------------------------------------------------------------
# Copyright (c) 2024, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

import argparse
import os
import platform
import struct
import ctypes
from pathlib import Path

if "Windows" in platform.system():
    import winreg


def _get_is_py64():
    return False if ((struct.calcsize("P") * 8) == 32) else True


class BinaryPathResolverWindows:
    def __init__(self):
        # NOTE: Don't allow for custom pathnames
        self._name_dll = "ArenaC_v140.dll"

        self._installation_root = None
        self._installation_pathname = None
        self._initialize_installation_vars()

    def _initialize_installation_vars(self):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Lucid Vision Labs\\Arena SDK") as key:
                # enable reflection if py32 ?
                # winreg.DisableReflectionKey(key)
                # winreg.EnableReflectionKey(key)

                root = winreg.QueryValueEx(key, "InstallFolder")[0]
                if root == "":
                    raise OSError
                else:
                    root = Path(root)
                    self._installation_root = root

                    if _get_is_py64():
                        self._installation_pathname = root / "x64Release" / self._name_dll
                    else:
                        self._installation_pathname = root / "Win32Release" / self._name_dll

        except OSError:
            self._installation_root = None
            self._installation_pathname = None


class BinaryPathResolverLinux:
    def __init__(self):
        # NOTE: Don't allow for custom pathnames
        self._name_so = "libarenac.so"

        self._installation_root = None
        self._installation_pathname = None
        self._initialize_installation_vars()

    def _initialize_installation_vars(self):
        # TODO change this to use ldconfig -v

        arena_sdk_conf_pathname = Path("/etc/ld.so.conf.d/Arena_SDK.conf")
        if not arena_sdk_conf_pathname.exists():
            self._installation_pathname = None
            return

        # check all paths in file and try to find the shared obj in it
        potential_paths = arena_sdk_conf_pathname.read_text().split()
        for path in potential_paths:
            pathname = Path(path) / self._name_so
            if pathname.exists():
                # because only 64 or 32 can be installed on the system.
                # if it was found then we return the first shared lib found
                self._installation_pathname = pathname
                if pathname.parent.parent.name.startswith("ArenaSDK"):
                    self._installation_root = pathname.parent.parent
                else:
                    raise OSError("Installation root does not start with 'ArenaSDK'")
                break


class BinaryPathResolver:
    def __init__(self):
        if "Windows" in platform.system():
            self._path_resolver = BinaryPathResolverWindows()

        elif "Linux" in platform.system():
            self._path_resolver = BinaryPathResolverLinux()

    def resolve(self):
        return self._path_resolver._installation_pathname

    def get_root(self):
        return self._path_resolver._installation_root

    def get_relative_root(self):
        return os.path.relpath(self._path_resolver._installation_root)

    def arena_sdk_found(self):
        return self._path_resolver._installation_root is not None


class BinaryLoaderWindows:
    def load(self, pathname):
        original_work_dir = os.getcwd()
        os.chdir(pathname.parent)
        handle = ctypes.CDLL(str(pathname))
        os.chdir(original_work_dir)
        return handle


def load_arena_sdk():
    if "Windows" in platform.system():
        resolver = BinaryPathResolver()
        pathname = resolver.resolve()
        return BinaryLoaderWindows().load(pathname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve binary paths for Arena SDK.")
    parser.add_argument("--relative", action="store_true", help="Print the relative root")
    parser.add_argument("--absolute", action="store_true", help="Print the absolute root")
    parser.add_argument("--found", action="store_true", help="Print True if Arena SDK is found")
    args = parser.parse_args()

    resolver = BinaryPathResolver()

    if args.relative:
        print(resolver.get_relative_root())
    elif args.absolute:
        print(resolver.get_root())
    elif args.found:
        print(resolver.arena_sdk_found())
