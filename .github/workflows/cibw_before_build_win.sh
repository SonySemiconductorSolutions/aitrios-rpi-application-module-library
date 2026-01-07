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


if command -v cmd.exe >/dev/null 2>&1; then
    echo "✅ Running on Windows or WSL"
else
    echo "❌ This install script only supports Windows"
    echo "System info:"
    uname -a
    exit 1
fi

# Check for an already installed Arena SDK
result=$(reg.exe query "HKLM\SOFTWARE\Lucid Vision Labs\Arena SDK" /v InstallFolder 2>/dev/null || true)
install_path=$(echo "$result" | grep "InstallFolder" | awk -F "REG_SZ" '{print $2}' | sed 's/^[[:space:]]*//')

if [ -n "$install_path" ]; then
    echo "Arena SDK is already installed at: $install_path"
else
    echo "Arena SDK not found; Installing Arena SDK."

    # Download Arena SDK
    ARENA_SDK_VERSION="v1.0.55.11"

    # Get ARENA_SDK_DOWNLOAD_URL from Windows environment if not already set
    if [ -z "${ARENA_SDK_DOWNLOAD_URL}" ]; then
        ARENA_SDK_DOWNLOAD_URL=$(cmd.exe /c "echo %ARENA_SDK_DOWNLOAD_URL%" 2>/dev/null | tr -d '\r\n' || echo "")
    fi

    # Download Arena SDK
    echo "Downloading Arena SDK..."
    curl -O "${ARENA_SDK_DOWNLOAD_URL}/ArenaSDK_${ARENA_SDK_VERSION}.exe"

    echo "Installing Arena SDK..."
    ./ArenaSDK_${ARENA_SDK_VERSION}.exe /quiet /norestart /install profile="Developer"
fi

# Install extra openssl dependency for Triton
openssl_path=$(where.exe openssl 2>/dev/null | head -n 1 || true)
if [ -n "$openssl_path" ]; then
    echo "OpenSSL is already installed at: $openssl_path"
else
    echo "OpenSSL not found; Installing OpenSSL..."
    choco install openssl -y
fi

# delvewheel is the equivalent of delocate/auditwheel for windows.
VENV_PATH="$(pwd)/.venv-delvewheel"
echo "Using Python virtual environment for delvewheel: $VENV_PATH"
test ! -d "$VENV_PATH" && python -m venv "$VENV_PATH"
. "$VENV_PATH/bin/activate"
python -m pip install delvewheel wheel