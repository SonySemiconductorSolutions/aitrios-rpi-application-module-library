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
import pytest
from unittest.mock import MagicMock, patch

from modlib.models import MODEL_TYPE

@pytest.fixture
def camera():
    with patch("modlib.devices.ai_camera.ai_camera.check_dir_required") as mock_check_dir_required:
        mock_check_dir_required.return_value = None
        from modlib.devices import AiCamera
        def init_overwrite(self, *args, **kwargs): 
            self.camera_id = None
        AiCamera.__init__ = init_overwrite
        yield AiCamera() 

@pytest.fixture
def model():
    # Create a mocked model with the necessary attributes
    model = MagicMock()
    model.model_file = "/foo/mock_model.onnx"
    model.model_type = "ONNX"
    model.color_format = "RGB"
    return model

@pytest.fixture
def mock_camera_methods(camera):
    camera.prepare_model_for_deployment = MagicMock()
    yield camera   

@pytest.fixture
def mock_dependencies(model, camera):
    with patch("os.path.exists", return_value=True) as mock_exists, \
         patch("modlib.devices.ai_camera.ai_camera.RPKPackager") as mock_packager_class, \
         patch("modlib.devices.ai_camera.ai_camera.IMX500Converter") as mock_converter_class:

        mock_packager = mock_packager_class.return_value
        mock_packager.run.return_value = None

        mock_converter = mock_converter_class.return_value
        mock_converter.run.return_value = None

        yield mock_converter, mock_packager, mock_exists, model, camera

def test_prepare_model_for_deployment_is_rpk(mock_dependencies):
    _, _, mock_exists, model, camera = mock_dependencies

    model.model_type = MODEL_TYPE.RPK_PACKAGED
    model.model_file = "/foo/network.rpk"

    mock_exists.return_value = True

    network_file = camera.prepare_model_for_deployment(model)

    assert network_file == model.model_file

def test_prepare_model_for_deployment_rpk_file_missing(mock_dependencies):
    _, _, mock_exists, model, camera = mock_dependencies

    model.model_type = MODEL_TYPE.RPK_PACKAGED
    model.model_file = "/foo/network.rpk"
    mock_exists.return_value = False

    network_file = camera.prepare_model_for_deployment(model)

    assert network_file is None

def test_prepare_model_for_deployment_is_converted(mock_dependencies):
    _, mock_packager, mock_exists, model, camera = mock_dependencies

    model.model_type = MODEL_TYPE.CONVERTED
    mock_exists.return_value = True
    
    network_file = camera.prepare_model_for_deployment(model)

    mock_packager.run.assert_called_once()
    assert network_file is not None

def test_prepare_model_for_deployment_is_converted_networkfile_missing(mock_dependencies): 
    _, mock_packager, mock_exists, model, camera = mock_dependencies
    
    model.model_type = MODEL_TYPE.CONVERTED
    mock_exists.return_value = False
    
    network_file = camera.prepare_model_for_deployment(model)

    mock_packager.run.assert_called_once()
    assert network_file is None

def test_prepare_model_for_deployment_is_framework(mock_dependencies):
    mock_converter, mock_packager, _, model, camera = mock_dependencies

    model.model_type = MODEL_TYPE.KERAS

    network_file = camera.prepare_model_for_deployment(model)

    mock_converter.run.assert_called_once()
    mock_packager.run.assert_called_once()
    assert network_file is not None

def test_prepare_model_for_deployment_is_framework_networkfile_missing(mock_dependencies):
    mock_converter, mock_packager, mock_exists, model, camera = mock_dependencies

    model.model_type = MODEL_TYPE.KERAS
    mock_exists.return_value = False

    network_file = camera.prepare_model_for_deployment(model)

    mock_converter.run.assert_called_once()
    mock_packager.run.assert_called_once()    
    assert network_file is None

def test_prepare_model_for_deployment_is_framework_onnx(mock_dependencies):
    mock_converter, mock_packager, _, model, camera = mock_dependencies

    model.model_type = MODEL_TYPE.ONNX

    network_file = camera.prepare_model_for_deployment(model)
    mock_converter.run.assert_called_once()
    mock_packager.run.assert_called_once()    

    assert network_file is not None

def test_prepare_model_for_deployment_is_framework_onnx_networkfile_missing(mock_dependencies):
    mock_converter, mock_packager, mock_exists, model, camera = mock_dependencies

    model.model_type = MODEL_TYPE.ONNX
    mock_exists.return_value = False

    network_file = camera.prepare_model_for_deployment(model)

    mock_converter.run.assert_called_once()
    mock_packager.run.assert_called_once()
    assert network_file is None

def test_deploy_networkfile(mock_camera_methods, model):   
    camera = mock_camera_methods

    with patch("modlib.devices.ai_camera.ai_camera.IMX500") as MockIMX500:
        camera.deploy(model)

        camera.prepare_model_for_deployment.assert_called_once()
        MockIMX500.assert_called_once_with(os.path.abspath(camera.prepare_model_for_deployment.return_value), camera_id=camera.camera_id)

def test_deploy_networkfile_missing(mock_camera_methods, model):
    camera = mock_camera_methods

    camera.prepare_model_for_deployment.return_value = None

    with pytest.raises(FileNotFoundError):
        camera.deploy(model)
