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

import os

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def download_imx500_rpk_model(model_file: str, save_dir: str) -> str | None:
    """
    Download a model file from the Raspberry Pi IMX500 models repository.

    Args:
        model_file: The name of the model file to download.
        save_dir: The local directory path where the downloaded file will be saved.

    Raises:
        RuntimeError: If the download fails due to a non-200 HTTP status code.

    Returns:
        The path to the downloaded model file.
    """

    RPI_IMX500_MODELS_URL = "https://github.com/raspberrypi/imx500-models/raw/main/"

    destination = os.path.join(save_dir, model_file)
    if os.path.exists(destination):
        return destination

    print(f"Downloading {model_file} from RPI IMX500 models repository...")

    # Create a session
    session = requests.Session()

    # Define a retry strategy
    retry_strategy = Retry(
        total=3,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1,
    )

    # Mount the retry strategy to the session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Use the session to make the request
    response = session.get(RPI_IMX500_MODELS_URL + model_file, timeout=10)

    if response.status_code == 200:
        os.makedirs(save_dir, exist_ok=True)
        with open(destination, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {model_file}.")
    else:
        raise RuntimeError(f"Failed to download file: {model_file}. Status code: {response.status_code}")

    return destination
