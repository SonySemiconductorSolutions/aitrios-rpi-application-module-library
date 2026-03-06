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
ONNX model inference (supports MCT quantized models).
"""
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import onnxruntime


def validate_onnx_model_path(model_path: str | Path) -> None:
    """
    Validate that the model path has a .onnx file extension.

    Args:
        model_path: Path to the model file (str or Path)

    Raises:
        ValueError: If the model path does not end with .onnx
    """
    model_path = Path(model_path)
    if model_path.suffix != ".onnx":
        raise ValueError(f"Model path expected to have .onnx extension, got: {model_path}")


def load_onnx_model(model_path: str, is_quantized: bool) -> onnxruntime.InferenceSession:
    """
    Loads the onnx model file.
    Requires onnx runtime and some other external modules to be installed.

    Raises:
        ImportError: When loading the model fails due to missing dependency.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        if is_quantized:
            import mct_quantizers as mctq  # noqa: F401

            # The following line is needed to make custom layers available for onnxruntime.InferenceSession
            from edgemdt_cl.pytorch import load_custom_ops
            so = load_custom_ops()
            so.log_severity_level = 3  # Suppress warnings

            model = onnxruntime.InferenceSession(
                model_path, sess_options=so, providers=["CPUExecutionProvider"]
            )

        else:
            model = onnxruntime.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )

        return model
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load onnx model: {e}")
    except Exception as e:
        raise Exception(f"Failed to load onnx model: {e}")


class Interpreter:
    def __init__(self, model_path: str, **opts: Any) -> None:
        """
        Initialize interpreter and load model.
        """
        validate_onnx_model_path(model_path)
        self.model_path = model_path
        self.error_msg = None

        # Load ONNX model
        self.onnx_model = load_onnx_model(model_path, opts.get("is_quantized", False))
        self.input_name = self.onnx_model.get_inputs()[0].name
        self.output_names = [x.name for x in self.onnx_model.get_outputs()]

    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        # Inference
        if self.onnx_model is None:
            raise RuntimeError("Failed to load ONNX model.")

        output_tensors = self.onnx_model.run(self.output_names, {self.input_name: input_tensor.astype(np.float32)})

        # Post-process (squeeze each tensor)
        squeezed_tensors = [np.squeeze(t) if t.ndim > 1 else t for t in output_tensors]

        # Return dict for .npz packing
        return {f"output{i}": arr for i, arr in enumerate(squeezed_tensors)}

    @property
    def input_tensor_size(self) -> Tuple[int, int]:
        # ONNX uses NCHW format: [batch, channels, height, width]
        # Return (width, height) to match the expected format in the codebase
        if self.onnx_model:
            shape = self.onnx_model.get_inputs()[0].shape
            return (shape[3], shape[2])  # (width, height)
        return None
