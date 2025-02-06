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

import glob
import os
import subprocess
from pathlib import Path
from typing import List


def run_shell_command(command: str):
    """
    Run shell command with output log and checking return code.

    Args:
        command: The shell command to be executed.

    Raises:
        subprocess.CalledProcessError: If the return code of the command is non-zero.
    """
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        shell=True,
        stderr=subprocess.STDOUT,
        text=True,
        close_fds=True,
    ) as process:
        for line in iter(process.stdout.readline, ""):
            print(line.rstrip())
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def check_dir_required(dir: Path, required_patterns: List[str]):
    """
    Check if a directory contains all the required files or directories based on the given patterns.

    Args:
        dir: The directory to check.
        required_patterns: A list of patterns (files or directories) that should be present.

    Raises:
        AssertionError: If the directory is not found at the specified path.
        AssertionError: If any of the required files or directories are missing in the directory.
    """
    assert os.path.isdir(dir), f"Directory not found at '{dir}'"

    # Check for each required file or directory pattern
    for pattern in required_patterns:
        # Find matching paths
        matching_paths = glob.glob(os.path.join(dir, pattern))
        assert matching_paths, f"Missing required file(s) or directory(ies) matching '{pattern}' in directory: {dir}"

        # Check if at least one of the matching paths is a file or a directory
        valid = any(os.path.isfile(path) or os.path.isdir(path) for path in matching_paths)
        assert valid, f"No file or directory found for pattern '{pattern}' in directory: {dir}"
