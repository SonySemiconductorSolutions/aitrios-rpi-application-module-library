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

import json
import logging
import os
import sys
import platform
from pathlib import Path
from typing import Optional

from modlib.devices.utils import run_shell_command
from modlib.models import COLOR_FORMAT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RPKPackager:
    """
    Helper class for packaging converted IMX500 models
    to a network `.rpk`-file accepted for an AiCamera deployment.

    Requires the installation of the imx500 packaging tools.
    Make sure to install them by running: `sudo apt install imx500-tools`.

    Example:
    ```
    packager = RPKPackager()
    packager.run(
        input_path="./path/to/packerOut.zip",
        output_dir="./pack",
        color_format=COLOR_FORMAT.RGB
    )
    ```
    """

    # NOTE: The RPK Packager includes the post-converter + packaging steps.

    def __init__(self):
        """
        Initialisation of the RPKPackager.
        """
        self.verified = False

    def _verify_rpk_packager(self):
        """
        Verifies initialisation of the RPKPackager.

        Raises:
            EnvironmentError: When the packager is initialised on a host other then a Raspberry Pi.
            FileNotFoundError: When the imx500 packaging tools are not installed.
        """

        if platform.system() != "Linux" or platform.machine() != "aarch64":
            raise EnvironmentError("This RPKPackager is intended to run on a Raspberry Pi.")

        self.packager_executable = "/usr/bin/imx500-package"

        if not os.path.isfile(self.packager_executable):
            raise FileNotFoundError(
                """
                The imx500-packaging tools not found.
                Please install it using `sudo apt install imx500-tools`.
            """
            )

        self.verified = True

    def run(
        self,
        input_path: Path,
        output_dir: Path,
        color_format: COLOR_FORMAT = COLOR_FORMAT.RGB,
        overwrite: Optional[bool] = None,
    ):
        """
        Packaging (post-converter + packager) using the locally installed packager.

        Args:
            input_path: The input file (packerOut.zip) to be packaged.
            output_dir: The directory where the packaged rpk file will be saved.
            color_format: Color format to package for. Defaults to `COLOR_FORMAT.RGB`.
            overwrite: If None, prompts the user for input. If True, overwrites the output directory if it exists.
                If False, terminates the packaging process if the directory exists.

        Raises:
            EnvironmentError: When the packager is initialised on a host other then a Raspberry Pi.
            FileNotFoundError: When the imx500 packaging tools are not installed.
        """
        # NOTE: only supports 1 network deployment for now (ordinal: 0)

        if isinstance(input_path, str):
            input_path = Path(input_path)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isfile(input_path) or not input_path.suffix == ".zip":
            raise FileNotFoundError("No converted zip-file found. Packager expects a `packerOut.zip` as input.")

        if os.path.exists(os.path.join(output_dir, "network.rpk")):
            if overwrite is None:
                user_input = input(
                    f"""
                    The given output directory '{output_dir}' already contains a `network.rpk` file.
                    1. Type 'yes/y' to overwrite the existing network.rpk file
                    2. Press <Enter> to use the already existing network.rpk file
                    3. Press 'no/n' to terminate the packaging process
                    Choice (y/<Enter>/n): """
                )
            else:
                if not isinstance(overwrite, bool):
                    raise ValueError("Invalid value for overwrite. It must be True or False.")
                user_input = "y" if overwrite else "<Enter>"

            if user_input.lower() in ("no", "n"):
                sys.exit()

            if user_input.lower() not in ("yes", "y"):
                logger.info("Model packaging terminated.")
                return

        if not self.verified:
            self._verify_rpk_packager()

        input_format = self.set_post_converter_config(output_dir, color_format)

        cmd = f"{self.packager_executable} -i {input_path} -o {output_dir} -f {input_format}"
        logger.info(f"Executing: {cmd}")
        run_shell_command(cmd)
        logger.info("Model packaging finished.")

    @staticmethod
    def set_post_converter_config(output_dir: Path, color_format: COLOR_FORMAT) -> Path:
        """
        Creates a JSON file in the output directory with the input tensor format configuration.
        Supported formats are "RGB", "BGR", "Y", or "BayerRGB".

        Args:
            output_dir: The directory where the JSON file will be saved.
            color_format: The color format to be used.
        """
        # NOTE: only supports 1 network deployment for now (ordinal: 0)

        if color_format not in ("RGB", "BGR", "Y", "BayerRGB"):
            raise ValueError(f"Unsupported color format: {color_format}. Supported formats are: RGB, BGR, Y, BayerRGB.")

        config = [{"ordinal": 0, "format": color_format}]

        output_file = output_dir / "input_format.json"
        with open(output_file, "w") as file:
            json.dump(config, file, indent=4)

        logger.info(f"Input tensor format JSON created at {output_file}")
        return output_file
