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
// #include <numpy/arrayobject.h>
#include <iostream>
#include <vector>

#include "posenet_decoder.h"
#include "posenet_wrapper.h"
#include "personlab_decoder.h"


// 1: Posenet
PyObject* decode_poses_cpp(PyObject* self, PyObject* args) {
    PyObject *score_obj, *shortOffset_obj, *middleOffset_obj;
    Py_buffer score_buf, shortOffset_buf, middleOffset_buf;

    // Parse Python objects (expecting buffer-capable objects like NumPy arrays)
    if (!PyArg_ParseTuple(args, "OOO", &score_obj, &shortOffset_obj, &middleOffset_obj)) {
        return NULL;
    }

    // Get raw buffer views (no conversion to Python lists, and avoiding copy)
    if (PyObject_GetBuffer(score_obj, &score_buf, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(shortOffset_obj, &shortOffset_buf, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(middleOffset_obj, &middleOffset_buf, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;

    // Access data as raw float arrays
    float* score = static_cast<float*>(score_buf.buf);
    float* shortOffset = static_cast<float*>(shortOffset_buf.buf);
    float* middleOffset = static_cast<float*>(middleOffset_buf.buf);

    // Use the data
    PosenetOutputDataType result;
    decode_poses(score, shortOffset, middleOffset, &result);

    // Release buffers (no need for delete[])
    PyBuffer_Release(&score_buf);
    PyBuffer_Release(&shortOffset_buf);
    PyBuffer_Release(&middleOffset_buf);

    // Convert result to Python objects
    PyObject* pose_scores_list = PyList_New(MAX_DETECTIONS);
    PyObject* pose_keypoints_list = PyList_New(result.n_detections);
    PyObject* pose_keypoint_scores_list = PyList_New(result.n_detections);

    for (int i = 0; i < MAX_DETECTIONS; ++i) {
        PyList_SetItem(pose_scores_list, i, PyFloat_FromDouble(result.pose_scores[i]));
    }

    for (int i = 0; i < result.n_detections; ++i) {
        PyObject* keypoints_sublist = PyList_New(NUM_KEYPOINTS);
        PyObject* keypoint_scores_sublist = PyList_New(NUM_KEYPOINTS);
        for (int j = 0; j < NUM_KEYPOINTS; ++j) {
            PyObject* keypoint = PyTuple_Pack(2, 
                PyFloat_FromDouble(result.pose_keypoints[i].keypoint[j].x),
                PyFloat_FromDouble(result.pose_keypoints[i].keypoint[j].y)
            );
            PyList_SetItem(keypoints_sublist, j, keypoint);
            PyList_SetItem(keypoint_scores_sublist, j, PyFloat_FromDouble(result.pose_keypoint_scores[i].keypoint[j]));
        }
        PyList_SetItem(pose_keypoints_list, i, keypoints_sublist);
        PyList_SetItem(pose_keypoint_scores_list, i, keypoint_scores_sublist);
    }

    return Py_BuildValue("iOOO", result.n_detections, pose_scores_list, pose_keypoints_list, pose_keypoint_scores_list);
}

// 2: Personlab
PyObject* decode_personlab_cpp(PyObject* self, PyObject* args) {
    PyObject *kp_maps_obj, *shortOffset_obj, *middleOffset_obj;
    Py_buffer kp_maps_buf, shortOffset_buf, middleOffset_buf;
    int num_keypoints;
    PyObject *edges_obj;
    float peak_threshold, nms_threshold, kp_radius;

    // Parse Python objects (expecting buffer-capable objects like NumPy arrays)
    if (!PyArg_ParseTuple(args, "OOOiOfff", &kp_maps_obj, &shortOffset_obj, &middleOffset_obj, 
                          &num_keypoints, &edges_obj, &peak_threshold, &nms_threshold, &kp_radius)) {
        return NULL;
    }

    // Get raw buffer views (no conversion to Python lists, and avoiding copy)
    if (PyObject_GetBuffer(kp_maps_obj, &kp_maps_buf, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(shortOffset_obj, &shortOffset_buf, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(middleOffset_obj, &middleOffset_buf, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;

    // Convert edges obj to std::vector<std::pair<int, int>>
    std::vector<std::pair<int, int>> edges;
    if (PyList_Check(edges_obj)) {
        for (Py_ssize_t i = 0; i < PyList_Size(edges_obj); ++i) {
            PyObject* edge = PyList_GetItem(edges_obj, i);
            if (PyTuple_Check(edge) && PyTuple_Size(edge) == 2) {
                int start = PyLong_AsLong(PyTuple_GetItem(edge, 0));
                int end = PyLong_AsLong(PyTuple_GetItem(edge, 1));
                edges.emplace_back(start, end);
            }
        }
    }

    // Use the data
    std::vector<std::vector<std::vector<float>>> skeletons;
    decode_personlab(
        static_cast<const float*>(kp_maps_buf.buf),
        static_cast<const float*>(shortOffset_buf.buf), 
        static_cast<const float*>(middleOffset_buf.buf),
        num_keypoints,
        edges,
        peak_threshold,
        nms_threshold,
        kp_radius,
        &skeletons
    );

    // Release buffers (no need for delete[])
    PyBuffer_Release(&kp_maps_buf);
    PyBuffer_Release(&shortOffset_buf);
    PyBuffer_Release(&middleOffset_buf);

    // Convert skeletons to a Python list
    PyObject* skeletons_list = PyList_New(skeletons.size());
    for (size_t i = 0; i < skeletons.size(); ++i) {
        PyObject* row_list = PyList_New(skeletons[i].size());
        for (size_t j = 0; j < skeletons[i].size(); ++j) {
            PyObject* col_list = PyList_New(skeletons[i][j].size());
            for (size_t k = 0; k < skeletons[i][j].size(); ++k) {
                PyList_SetItem(col_list, k, PyFloat_FromDouble(skeletons[i][j][k]));
            }
            PyList_SetItem(row_list, j, col_list);
        }
        PyList_SetItem(skeletons_list, i, row_list);
    }

    return Py_BuildValue("O", skeletons_list);
}

// Exposed methods definitions
static PyMethodDef methods[] = {
    {"decode_poses_cpp", (PyCFunction)decode_poses_cpp, METH_VARARGS, NULL},
    {"decode_personlab_cpp", (PyCFunction)decode_personlab_cpp, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL},
};

// Module: cpp_post_processors
// int init_numpy() {
//     import_array();
//     return 0;
// }

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "cpp_post_processors",
    NULL,
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_cpp_post_processors(void) {
    PyObject* m = PyModule_Create(&module);
    if (!m) return NULL;

    // // Ensure NumPy initializes properly
    // if (init_numpy() < 0) { 
    //     Py_DECREF(m);
    //     return NULL;
    // }

    return m;
}