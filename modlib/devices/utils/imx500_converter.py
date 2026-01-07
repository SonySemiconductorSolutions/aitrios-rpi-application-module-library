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

import logging
import os
import platform
import sys
from pathlib import Path
from typing import List, Optional

from modlib.models import MODEL_TYPE

from .common import run_shell_command

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IMX500Converter:
    """
    Helper class for converting a Model to IMX500 accepted format.
    Converts a KERAS or ONNX model to a `packerOut.zip` file ready for packaging.

    Required dependencies are automatically installed into a dedicated virtual environment and depent on its model type:
    - KERAS models: require `imx500-converter[tf]` and is automatically installed in `~/.modlib/.venv-imx500-converter-tf`
    - ONNX models: require `imx500-converter[pt]` and is automatically installed in `~/.modlib/.venv-imx500-converter-pt`

    Example:
    ```
    converter = IMX500Converter()
    converter.run(
        model_file="./path/to/model.keras",
        model_type=MODEL_TYPE.KERAS,
        output_dir="./pack"
    )
    ```
    """

    def __init__(self):
        """
        Initialisation of the IMX500Converter.
        """
        self.root = os.getenv("MODLIB_HOME", os.path.expanduser("~/.modlib"))
        os.makedirs(self.root, exist_ok=True)

    def run(self, model_file: Path, model_type: MODEL_TYPE, output_dir: Path, overwrite: Optional[bool] = None):
        """
        Run the converter in its dedicated virtual environment corresponding to the model type.

        Args:
            model_file: Path to the model file to be converted.
            model_type: Model type of the provided model to be converted.
            output_dir: The directory where the converted `packerOut.zip` file will be saved.
            overwrite: If None, prompts the user for input. If True, overwrites the output directory if it exists.
                If False, terminates the conversion/packaging process if the directory exists.
        """
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"The model file '{model_file}' does not exist.")

        # NOTE: Keep in mind that currently only ONNX models coming from a pytorch framework model are supported.
        # The current implementation of the imx500-converter requires the tf dependencies for converting keras models
        # and the pt dependencies for converting ONNX models.

        if model_type == MODEL_TYPE.KERAS:
            converter_executable = self._get_converter_executable(
                env_name=".venv-imx500-converter-tf", requirements=["imx500-converter[tf]"], imxconv_name="imxconv-tf"
            )
        elif model_type == MODEL_TYPE.ONNX:
            converter_executable = self._get_converter_executable(
                env_name=".venv-imx500-converter-pt", requirements=["imx500-converter[pt]"], imxconv_name="imxconv-pt"
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        flag = ""
        if os.path.exists(os.path.join(output_dir, "packerOut.zip")):
            if overwrite is None:
                user_input = input(
                    f"""
                    The given output directory '{output_dir}' already contains a `packerOut.zip` file.
                    1. Type 'yes/y' to recompile and overwrite the existing packerOut.zip file
                    2. Press <Enter> to use the already existing packerOut.zip file
                    3. Press 'no/n' to terminate the conversion process
                    Choice (y/<Enter>/n): """
                )
            else:
                if not isinstance(overwrite, bool):
                    raise ValueError("Invalid value for overwrite. It must be True or False.")
                user_input = "y" if overwrite else "<Enter>"

            if user_input.lower() in ("no", "n"):
                sys.exit()

            if user_input.lower() not in ("yes", "y"):
                logger.info("Model conversion terminated.")
                return

            flag = "--overwrite-output"

        # Executing IMX500 model conversion
        cmd = f"{converter_executable} -i {model_file} -o {output_dir} {flag}"
        logger.info(f"Executing: {cmd}")
        run_shell_command(cmd)
        logger.info("Model conversion finished.")

    def _get_converter_executable(self, env_name, requirements, imxconv_name) -> str:
        # Check and possibly create virtual environment for the converter
        env_path = self.__create_virtualenv(self.root, env_name, requirements)

        converter_executable = os.path.join(
            env_path, "Scripts" if platform.system() == "Windows" else "bin", imxconv_name
        )

        if not os.path.exists(converter_executable):
            raise FileNotFoundError(f"Converter executable not found at {converter_executable}")

        return converter_executable

    @staticmethod
    def __create_virtualenv(root: str, env_name: str, requirements: List[str]) -> str:
        if not os.path.exists(root) or not os.path.isdir(root):
            raise FileNotFoundError(f"The root directory '{root}' does not exist.")
        if not isinstance(env_name, str):
            raise TypeError("env_name must be a string")
        if not all(isinstance(item, str) for item in requirements):
            raise TypeError("requirements must be a list of strings")

        env_path = os.path.join(root, env_name)
        if not os.path.exists(env_path):
            logger.info(f"Converter: Creating virtual environment at {env_path}")
            run_shell_command(f"{sys.executable} -m venv {env_path}")

            pip_executable = os.path.join(env_path, "Scripts" if platform.system() == "Windows" else "bin", "pip")

            for package in requirements:
                run_shell_command(f"{pip_executable} install {package}")

        return env_path
