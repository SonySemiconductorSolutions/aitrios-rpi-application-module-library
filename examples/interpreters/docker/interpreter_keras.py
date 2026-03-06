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
TensorFlow Keras model inference (supports MCT quantized models).
"""
from pathlib import Path
import numpy as np
import json
import zipfile
from typing import Any, Dict, Tuple
import tensorflow as tf
try:
    import keras  # standalone keras
except ImportError:  # pragma: no cover
    keras = None
# ---------------------------------------------------------------------
# Model loading (TensorFlow / MCT quantized)
# ---------------------------------------------------------------------
def validate_keras_model_path(model_path: str | Path) -> None:
    """
    Validate that the model path has a .keras file extension.

    Args:
        model_path: Path to the model file (str or Path)

    Raises:
        ValueError: If the model path does not end with .keras
    """
    model_path = Path(model_path)
    if model_path.suffix != ".keras":
        raise ValueError(f"Model path expected to have .keras extension, got: {model_path}")


def get_saved_keras_version(model_path: str | Path) -> str | None:
    """
    Extract the keras_version field from a .keras model file.
    Returns None if not found.
    """
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    try:
        with zipfile.ZipFile(model_path, "r") as z:
            names = set(z.namelist())

            # New Keras v3 format: metadata.json
            meta_name = None
            if "metadata.json" in names:
                meta_name = "metadata.json"
            elif "keras_metadata.json" in names:  # older nightlies / variants
                meta_name = "keras_metadata.json"

            if meta_name is not None:
                with z.open(meta_name) as f:
                    metadata = json.load(f)
                # usual key is "keras_version"
                kv = metadata.get("keras_version")
                if isinstance(kv, str):
                    return kv
                # some variants store nested dict
                if isinstance(kv, dict) and "version" in kv:
                    return kv["version"]

            # Fallback: older layouts might store it in config.json
            if "config.json" in names:
                with z.open("config.json") as f:
                    cfg = json.load(f)
                kv = cfg.get("keras_version")
                if isinstance(kv, str):
                    return kv
    except:
        pass

    return None


def compare_versions(model_path: str | Path) -> tuple[bool, str | None, str | None]:
    """
    Compare saved model version with installed TensorFlow/Keras version.

    Returns:
        Tuple of (is_compatible, saved_version, installed_version)
    """
    saved = get_saved_keras_version(model_path)

    current_tf = tf.__version__
    current_keras = getattr(keras, "__version__", None) if keras is not None else None

    print(f"Model file     : {model_path}")
    print(f"Saved Keras ver: {saved}")
    print(f"TF installed   : {current_tf}")
    if current_keras is not None:
        print(f"Keras installed: {current_keras}")
    else:
        print("Keras installed: <standalone keras not installed>")

    if saved is None:
        print("⚠ Could not find keras_version in the model file.")
        print("⚠ Assuming compatible - will attempt to load model.")
        return (True, None, current_tf)  # Assume compatible if we can't determine version

    # Compare against whatever you consider authoritative
    target = current_keras or current_tf
    if target is None:
        print("⚠ No local Keras/TensorFlow version available for comparison.")
        print("⚠ Assuming compatible - will attempt to load model.")
        return (True, saved, None)  # Assume compatible if we can't determine installed version

    is_compatible = saved == target
    if is_compatible:
        print("✔ Saved Keras version matches installed version.")
    else:
        print(f"⚠ Version mismatch: saved={saved}, installed={target}")

    return (is_compatible, saved, target)


def load_older(model_path: str, use_mct: bool = False):
    """Primarily handling models saved w tf<2.15

    sony-custom-layers seems to handle custom ops like
    - SDD post process
    - etc...

    mct.keras_load_quantized_model handles layers like
    - KerasActivationQuantizationHolder
    - KerasQuantizationWrapper
    - etc...

    use_mct=False removes mct dependency,
      but have to handle custom layers manually

    Notes:
    - you can choose to use mct or tensorflow to load model,
      with only tensorflow you skip mct dependency,
      but have to handle custom layers manually
    - nanodet can be loaded without custom_layers_scope
      instead of ssd post process, it uses combined_non_max_supression
    - ssd requires custom_layers_scope, probably because of the ssd post process

    # requires-python = ">=3.11"
    # dependencies = [
    #   "tensorflow==2.13.0",
    #   "numpy",
    #   "mct-quantizers",
    #   "sony-custom-layers",  #edge-mdt-cl requires tf>=2.15
    # ]
    """
    # sony_custom_layers is deprecated and replaced by edge-mdt-cl
    # but we use it here for models saved w tf<2.15
    from sony_custom_layers.keras import custom_layers_scope

    print(f"[load_older] Loading model: {model_path}, use_mct={use_mct}")

    with custom_layers_scope():
        if use_mct:
            import model_compression_toolkit as mct
            model = mct.keras_load_quantized_model(model_path)
        else:
            import tensorflow as tf
            from mct_quantizers import (
                KerasActivationQuantizationHolder,
                KerasQuantizationWrapper
            )
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={"KerasActivationQuantizationHolder": KerasActivationQuantizationHolder,
                                "KerasQuantizationWrapper": KerasQuantizationWrapper}
                )
    return model

def load_tf_keras_model(model_path: str, is_quantized: bool):
    """
    Load a Keras model file using TensorFlow or model_compression_toolkit (if quantized).
    """
    try:
        print(f"Loading model: {model_path}, is_quantized={is_quantized}")
        if is_quantized:
            return load_older(model_path, use_mct=False)
        else:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load Keras model: {e}")
    except Exception as e:
        raise Exception(f"Failed to load Keras model: {e}")


# ---------------------------------------------------------------------
# Interpreter wrapper
# ---------------------------------------------------------------------
class Interpreter:
    def __init__(self, model_path: str, **opts: Any) -> None:
        """
        Initialize interpreter and load model.
        Sets self.error_msg if there's a compatibility issue, None if successful.
        """
        validate_keras_model_path(model_path)
        self.model_path = model_path
        self.error_msg = None
        is_compatible, saved_version, installed_version = compare_versions(model_path)

        if not is_compatible:
            self.error_msg = f"Model version incompatible: model saved with Keras {saved_version}, but TensorFlow {installed_version} is installed"
            print(f"[Interpreter] {self.error_msg}")
            self.keras_model = None
        else:
            self.keras_model = load_tf_keras_model(model_path, opts.get("is_quantized", False))

    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        # Inference
        if self.keras_model is None:
            raise RuntimeError("Model is not compatible with the installed version of TensorFlow.")

        output_tensors = self.keras_model.predict(input_tensor, verbose=0)

        # Post-process (squeeze each tensor)
        squeezed_tensors = [np.squeeze(t) if t.ndim > 1 else t for t in output_tensors]

        # Return dict for .npz packing
        return {f"output{i}": arr for i, arr in enumerate(squeezed_tensors)}

    @property
    def input_tensor_size(self) -> Tuple[int, int]:
        # Keras uses NHWC format: (batch, height, width, channels)
        # We need to return (width, height) to match the expected format
        if self.keras_model:
            shape = self.keras_model.input_shape
            return (shape[2], shape[1])  # (width, height)
        return None
