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

SRC_CPP := posenet_binding.cpp posenet_decoder.cpp

CXXFLAGS := \
	-W \
	-Wall \
	-Wextra \
	-O3 \
	-std=c++17 \
	-shared \
	-I . \

# Detect Windows
ifeq ($(OS),Windows_NT)
   	ARCH := $(PROCESSOR_ARCHITEW6432)
    ifeq ($(ARCH),AMD64)
        TARGET := libposenet_win64.dll
    else
        $(error Unsupported architecture: $(ARCH))
    endif
	CXX := C:/mingw64/bin/g++.exe
	CXXFLAGS += -Wl,--add-stdcall-alias
	CXXFLAGS += -static
else
    ARCH := $(shell dpkg --print-architecture)
    ifeq ($(filter $(ARCH),amd64 arm64),)
        $(error Unsupported architecture: $(ARCH))
    endif
    TARGET := libposenet_$(ARCH).so
	CXX	:= g++
	CXXFLAGS += -fPIC
endif


all: clean $(TARGET)

$(TARGET): $(SRC_CPP)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	if [ -f $(TARGET) ]; then rm -f $(TARGET); fi