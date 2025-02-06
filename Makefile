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
PYTHON3 := python3
ifeq ($(OS),Windows_NT)
    pyact := . .venv/Scripts/activate
else
    pyact := . .venv/bin/activate
endif

.PHONY: docs


setup: .venv

.venv:
	test -d .venv || $(PYTHON3) -m venv .venv --system-site-packages
	$(pyact); \
	python -m pip install --upgrade pip; \
	pip install --ignore-installed poetry; \
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring; \
	poetry install

clean: clean-pyc
	rm -rf .venv
	rm -f poetry.lock
	rm -rf build
	rm -f gui-tool.spec

clean-pyc:
	find . -name "__pycache__" -exec rm -fr {} +
	find . -name ".pytest_cache" -exec rm -fr {} +
	find . -name ".coverage" -exec rm -fr {} +

test: .venv
	$(pyact); \
	python -m pytest --junitxml=pytest.xml -m 'not slow and not aicam'
	make clean-pyc > /dev/null

lint: .venv
	$(pyact); \
	isort .; \
	black .; \
	flake8

# BUILD
MODLIB_VERSION ?= 1.0.0
build: .venv
	$(pyact); \
	poetry version $(MODLIB_VERSION); \
	poetry build