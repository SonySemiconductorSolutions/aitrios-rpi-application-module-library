#!/bin/bash
set -e

#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# Check if running on Linux
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "❌ This install script only supports Linux." >&2
  exit 1
fi

arch=$(uname -m)

# Check if architecture is supported
if [[ "$arch" == "x86_64" ]]; then
  echo "✅ Running on x86_64 (64-bit Intel/AMD)"
  ARENA_SDK_VERSION="v0.1.95"
  ARCH="x64"
elif [[ "$arch" == "aarch64" ]]; then
  echo "✅ Running on ARM64 (aarch64)"
  ARENA_SDK_VERSION="v0.1.78"
  ARCH="ARM64"
else
  echo "❌ Unsupported architecture: $arch" >&2
  exit 1
fi

# Download Arena SDK
curl -O "${ARENA_SDK_DOWNLOAD_URL}/ArenaSDK_${ARENA_SDK_VERSION}_Linux_${ARCH}.tar.gz"

# Extract Arena SDK
tar -xvzf "ArenaSDK_${ARENA_SDK_VERSION}_Linux_${ARCH}.tar.gz" -C /tmp
cd /tmp/ArenaSDK_Linux_${ARCH}

MIDDLE=$([[ "$arch" == "x86_64" ]] && echo "Linux_" || echo "")
OG_CONF_FILE="Arena_SDK_${MIDDLE}${ARCH}.conf"
MOD_CONF_FILE="Arena_SDK_${ARCH}_modified.conf"

# Install Arena SDK
# in a manylinux container (x86_64), just replace the apt-get line and adapt the sh -c lines
# (on aarch64 no apt-get installs are needed, so the conf file is not modified)
dnf update -y
sed 's/sh -c "\(.*\)"/\1/g' ${OG_CONF_FILE} > ${MOD_CONF_FILE}
sed -i 's/sudo apt-get -y install libibverbs1 librdmacm1/dnf -y install libibverbs librdmacm/g' ${MOD_CONF_FILE}
source ${MOD_CONF_FILE}

# Verify installation by checking if libarenac.so is installed in the correct location
lib_path=$(ldconfig -p | grep -F "libarenac.so " | awk -F'=> ' '{print $2}' | xargs)
if [[ ! "$lib_path" == /tmp/ArenaSDK_Linux_${ARCH}/* ]]; then
    echo "Error: Arena SDK library not found in expected location"
    exit 1
fi

# Install extra openssl dependency for Triton
dnf install -y openssl-devel

echo "Arena SDK installed successfully"
