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

.ONESHELL:
SHELL := /bin/bash

ifeq (,$(shell which uv || echo $(OS) | grep -i Windows_NT))
 	$(error "uv notfound. To install uv, run the following commands: curl -LsSf https://astral.sh/uv/install.sh | sh")
endif

setup:
	uv sync

# NOTE: when numpy include over the meson build you have to disable build isolation
# setup:
# 	uv venv
# 	uv pip install "meson-python~=0.17.1" "numpy>=2.0.0rc1,<=2.2.0"
# 	UV_NO_BUILD_ISOLATION=true uv pip install -e .

clean: clean-pyc
	rm -rf .venv .ruff_cache
	rm -rf modlib.egg-info dist pytest.xml

clean-pyc:
	find . -name "__pycache__" -exec rm -fr {} +
	find . -name ".pytest_cache" -exec rm -fr {} +
	find . -name ".coverage" -exec rm -fr {} +

test:
	uv run pytest --junitxml=pytest.xml -m 'not slow and not aicam'
	make clean-pyc > /dev/null

lint:
	uv run ruff format ./modlib 
	uv run ruff check --fix ./modlib 

build:
	uv build
