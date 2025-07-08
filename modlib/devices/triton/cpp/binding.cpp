/*
 * Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <Python.h>

#ifdef ARENA_SDK_FOUND
#include "triton.h"
#endif

#include <iostream>
#include <vector>
#include <stdexcept>


// Check if Arena SDK is found
// Always expose this function to Python

PyObject* arena_sdk_found(PyObject* self, PyObject* args) {
    #ifdef ARENA_SDK_FOUND
    return Py_BuildValue("i", 1);
    #else
    return Py_BuildValue("i", 0);
    #endif
}

// Only expose the following functions if Arena SDK is found
#ifdef ARENA_SDK_FOUND

PyObject* start(PyObject* self, PyObject* args) {
    PyObject* py_callback;

    if (!PyArg_ParseTuple(args, "O", &py_callback)) {
        return NULL;
    }

    // Check if the callback is callable
    if (!PyCallable_Check(py_callback)) {
        PyErr_SetString(PyExc_TypeError, "py_callback must be callable");
        return NULL;
    }

    // Call function with the callback
    start_triton(py_callback);

    // Return None
    Py_RETURN_NONE;
}


PyObject* upload_file(PyObject* self, PyObject* args) {
    const char* filename;
    int fileType;
    if (!PyArg_ParseTuple(args, "si", &filename, &fileType)) {
        return NULL;
    }

    DATNetworkInfo result = triton_upload_file(filename, fileType);

    if (!result.is_dat_info) {
        Py_RETURN_NONE;
    }

    // Return results
    PyObject* norm_val_list = Py_BuildValue("[iii]", result.norm_val[0], result.norm_val[1], result.norm_val[2]);
    PyObject* norm_shift_list = Py_BuildValue("[iii]", result.norm_shift[0], result.norm_shift[1], result.norm_shift[2]);
    PyObject* div_val_list = Py_BuildValue("[iii]", result.div_val[0], result.div_val[1], result.div_val[2]);
    PyObject* input_tensor_dict = Py_BuildValue("{s:O,s:O,s:O,s:i}",
                                          "norm_val", norm_val_list,
                                          "norm_shift", norm_shift_list,
                                          "div_val", div_val_list,
                                          "div_shift", result.div_shift);

    PyObject* result_dict = Py_BuildValue("{s:O}", "input_tensor", input_tensor_dict);
    return result_dict;
}


PyObject* test_connection(PyObject* self, PyObject* args) {
    simple_acquisition();
    Py_RETURN_NONE;
}

#endif


// Exposed methods definitions
static PyMethodDef methods[] = {
    {"arena_sdk_found", (PyCFunction)arena_sdk_found, METH_VARARGS, NULL},
#ifdef ARENA_SDK_FOUND
    {"start", (PyCFunction)start, METH_VARARGS, NULL},
    {"upload_file", (PyCFunction)upload_file, METH_VARARGS, NULL},
    {"test_connection", (PyCFunction)test_connection, METH_VARARGS, NULL},
#endif
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "triton_cpp",
    NULL,
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_triton_cpp(void) {
    PyObject* m = PyModule_Create(&module);
    if (!m) return NULL;

    return m;
}