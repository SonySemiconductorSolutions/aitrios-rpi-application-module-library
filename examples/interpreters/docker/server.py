#
# Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.
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

"""
Docker-based FastAPI server for TensorFlow Keras model inference (supports MCT quantized models).
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, HTTPException, Response
from pydantic import BaseModel
from uuid import uuid4
from io import BytesIO


# Import Interpreter based on environment variable
_backend = os.getenv("BACKEND", "").lower()
if _backend == "onnx":
    from interpreter_onnx import Interpreter
elif _backend in ["keras-tf2.13", "keras-tf2.14"]:
    from interpreter_keras import Interpreter
else:
    raise ValueError(f"Invalid BACKEND: {_backend}. Must be 'onnx', 'keras-tf2.13', or 'keras-tf2.14'")


# ---------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------
app = FastAPI()
_sessions: Dict[str, Interpreter] = {}


class InitReq(BaseModel):
    model_uri: str
    options: Dict[str, Any] | None = None  # e.g. {"is_quantized": True}


@app.post("/init")
def init(req: InitReq) -> Dict[str, Any]:
    opts = req.options or {}

    # validate existence inside the container
    model_path = Path(req.model_uri)
    if not model_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Model path not found: {model_path}"
        )
    if not model_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Model path is not a file: {model_path}"
        )

    sid = uuid4().hex
    interpreter = Interpreter(req.model_uri, **opts)
    _sessions[sid] = interpreter

    # Return error message from interpreter if any
    response = {"session_id": sid}
    if interpreter.error_msg:
        response["status"] = "error"
        response["error"] = interpreter.error_msg
    else:
        response["status"] = "success"

    return response


@app.delete("/session/{session_id}")
def close(session_id: str) -> Dict[str, str]:
    if session_id in _sessions:
        _sessions.pop(session_id)
        return {"status": "closed"}
    raise HTTPException(404, "unknown session_id")


@app.get("/input_tensor_size/{session_id}")
def get_input_tensor_size(session_id: str) -> Response:
    if session_id not in _sessions:
        raise HTTPException(404, "unknown session_id")
    payload = {"input_tensor_size": _sessions[session_id].input_tensor_size}
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
    )


@app.post("/infer/{session_id}")
async def infer(session_id: str, input_npy: UploadFile) -> Response:
    if session_id not in _sessions:
        raise HTTPException(404, "unknown session_id")

    raw = await input_npy.read()
    x = np.load(BytesIO(raw), allow_pickle=False)  # expects .npy payload

    outs = _sessions[session_id].infer(x)

    # Serialize outputs → .npz bytes
    buf = BytesIO()
    np.savez(buf, **outs)
    payload = buf.getvalue()

    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "Content-Type": "application/x.numpy-npz",
            "Content-Disposition": 'attachment; filename="outputs.npz"',
            "X-Tensor-Names": ",".join(outs.keys()),
        },
    )
