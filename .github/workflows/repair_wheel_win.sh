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


echo "paths..."
WHEEL=$(wslpath "$1")
DEST_DIR=$(wslpath "$2")

# Repair wheel
VENV_PATH="$(pwd)/.venv-delvewheel"
test -d "$VENV_PATH" || { echo "❌ Virtual environment directory not found at $VENV_PATH"; exit 1; }
. "$VENV_PATH/bin/activate"

echo "Showing wheel..."
delvewheel show $WHEEL

echo "Repairing wheel..."
delvewheel repair --no-mangle-all -w $DEST_DIR $WHEEL
