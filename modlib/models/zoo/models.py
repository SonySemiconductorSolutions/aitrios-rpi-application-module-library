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
from typing import List

import numpy as np

from ..model import COLOR_FORMAT, MODEL_TYPE, Model
from ..post_processors import (
    pp_cls,
    pp_cls_softmax,
    pp_higherhrnet,
    pp_od_bscn,
    pp_od_efficientdet_lite0,
    pp_posenet,
    pp_segment,
    pp_od_yolo_ultralytics,
)
from ..results import Classifications, Detections, Poses, Segments
from .utils import download_imx500_rpk_model

ASSETS_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/assets"
ZOO_DIR = f"{os.getenv('MODLIB_HOME', os.path.expanduser('~/.modlib'))}/zoo"


# NOTE: The current model zoo implementation is not generic and not compatible with other devices.
#       The model zoo is currently tailored to the Raspberry Pi IMX500 AI Camera, and will by default
#       download RPK packaged models only.


# ###############################################################################
# ############################# Input Tensor Only ###############################
# ###############################################################################


class InputTensorOnly(Model):
    """
    ```
    from modlib.models.zoo import InputTensorOnly
    model = InputTensorOnly()
    ```
    The network file `imx500_network_inputtensoronly.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_inputtensoronly.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        return image

    def post_process(self, output_tensors: List[np.ndarray]):
        raise ValueError("No output tensors to process for InputTensorOnly model.")


# ###############################################################################
# ########################### Classification models #############################
# ###############################################################################


class EfficientNetB0(Model):
    """
    ```
    from modlib.models.zoo import EfficientNetB0
    model = EfficientNetB0()
    ```
    The network file `imx500_network_efficientnet_bo.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_efficientnet_bo.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class EfficientNetLite0(Model):
    """
    ```
    from modlib.models.zoo import EfficientNetLite0
    model = EfficientNetLite0()
    ```
    The network file `imx500_network_efficientnet_lite0.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_efficientnet_lite0.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class EfficientNetV2B0(Model):
    """
    ```
    from modlib.models.zoo import EfficientNetV2B0
    model = EfficientNetV2B0()
    ```
    The network file `imx500_network_efficientnetv2_b0.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_efficientnetv2_b0.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls(output_tensors)


class EfficientNetV2B1(Model):
    """
    ```
    from modlib.models.zoo import EfficientNetV2B1
    model = EfficientNetV2B1()
    ```
    The network file `imx500_network_efficientnetv2_b1.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_efficientnetv2_b1.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls(output_tensors)


class EfficientNetV2B2(Model):
    """
    ```
    from modlib.models.zoo import EfficientNetV2B2
    model = EfficientNetV2B2()
    ```
    The network file `imx500_network_efficientnetv2_b2.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_efficientnetv2_b2.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls(output_tensors)


class MNASNet1_0(Model):
    """
    ```
    from modlib.models.zoo import MNASNet1_0
    model = MNASNet1_0()
    ```
    The network file `imx500_network_mnasnet1.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_mnasnet1.0.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class MobileNetV2(Model):
    """
    ```
    from modlib.models.zoo import MobileNetV2
    model = MobileNetV2()
    ```
    The network file `imx500_network_mobilenet_v2.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_mobilenet_v2.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls(output_tensors)


class MobileViTXS(Model):
    """
    ```
    from modlib.models.zoo import MobileViTXS
    model = MobileViTXS()
    ```
    The network file `imx500_network_mobilevit_xs.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_mobilevit_xs.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.BGR,
            preserve_aspect_ratio=True,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class MobileViTXXS(Model):
    """
    ```
    from modlib.models.zoo import MobileViTXXS
    model = MobileViTXXS()
    ```
    The network file `imx500_network_mobilevit_xxs.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_mobilevit_xxs.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class RegNetX002(Model):
    """
    ```
    from modlib.models.zoo import RegNetX002
    model = RegNetX002()
    ```
    The network file `imx500_network_regnetx_002.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_regnetx_002.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class RegNetY002(Model):
    """
    ```
    from modlib.models.zoo import RegNetY002
    model = RegNetY002()
    ```
    The network file `imx500_network_regnety_002.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_regnety_002.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class RegNetY004(Model):
    """
    ```
    from modlib.models.zoo import RegNetY004
    model = RegNetY004()
    ```
    The network file `imx500_network_regnety_004.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_regnety_004.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class ResNet18(Model):
    """
    ```
    from modlib.models.zoo import ResNet18
    model = ResNet18()
    ```
    The network file `imx500_network_resnet18.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_resnet18.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class ShuffleNetV2X1_5(Model):
    """
    ```
    from modlib.models.zoo import ShuffleNetV2X1_5
    model = ShuffleNetV2X1_5()
    ```
    The network file `imx500_network_shufflenet_v2_x1_5.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_shufflenet_v2_x1_5.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


class SqueezeNet1_0(Model):
    """
    ```
    from modlib.models.zoo import SqueezeNet1_0
    model = SqueezeNet1_0()
    ```
    The network file `imx500_network_squeezenet1.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_squeezenet1.0.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/imagenet_labels.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        return pp_cls_softmax(output_tensors)


# ###############################################################################
# ########################## Object Detection models ############################
# ###############################################################################


class EfficientDetLite0(Model):
    """
    ```
    from modlib.models.zoo import EfficientDetLite0
    model = EfficientDetLite0()
    ```
    The network file `imx500_network_efficientdet_lite0_pp.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(
                model_file="imx500_network_efficientdet_lite0_pp.rpk", save_dir=ZOO_DIR
            ),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=True,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/coco_labels_91.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_efficientdet_lite0(output_tensors)


class NanoDetPlus416x416(Model):
    """
    ```
    from modlib.models.zoo import NanoDetPlus416x416
    model = NanoDetPlus416x416()
    ```
    The network file `imx500_network_nanodet_plus_416x416_pp.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(
                model_file="imx500_network_nanodet_plus_416x416_pp.rpk", save_dir=ZOO_DIR
            ),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.BGR,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/coco_labels_80.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_bscn(output_tensors)


class SSDMobileNetV2FPNLite320x320(Model):
    """
    ```
    from modlib.models.zoo import SSDMobileNetV2FPNLite320x320
    model = SSDMobileNetV2FPNLite320x320()
    ```
    The network file `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(
                model_file="imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk", save_dir=ZOO_DIR
            ),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/coco_labels_91.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_bscn(output_tensors)


class YOLOv8n(Model):
    """
    ```
    from modlib.models.zoo import YOLOv8n
    model = YOLOv8n()
    ```
    The network file `imx500_network_yolov8n_pp.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_yolov8n_pp.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/coco_labels_80.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_yolo_ultralytics(output_tensors)


class YOLO11n(Model):
    """
    ```
    from modlib.models.zoo import YOLO11n
    model = YOLO11n()
    ```
    The network file `imx500_network_yolo11n_pp.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_yolo11n_pp.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/coco_labels_80.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_yolo_ultralytics(output_tensors)


# ###############################################################################
# ########################### Pose Estimation models ############################
# ###############################################################################


class Posenet(Model):
    """
    ```
    from modlib.models.zoo import Posenet
    model = Posenet()
    ```
    The network file `imx500_network_posenet.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_posenet.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Poses:
        return pp_posenet(output_tensors)


class HigherHRNet(Model):
    """
    ```
    from modlib.models.zoo import HigherHRNet
    model = HigherHRNet()
    ```
    The network file `imx500_network_higherhrnet_coco.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_higherhrnet_coco.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Poses:
        return pp_higherhrnet(output_tensors)


# ###############################################################################
# ############################ Segmentation models ##############################
# ###############################################################################


class DeepLabV3Plus(Model):
    """
    ```
    from modlib.models.zoo import DeepLabV3Plus
    model = DeepLabV3Plus()
    ```
    The network file `imx500_network_deeplabv3plus.rpk` is downloaded from
    the [Raspberry Pi Model Zoo](https://github.com/raspberrypi/imx500-models)
    """

    def __init__(self):
        super().__init__(
            model_file=download_imx500_rpk_model(model_file="imx500_network_deeplabv3plus.rpk", save_dir=ZOO_DIR),
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(f"{ASSETS_DIR}/pascal_voc_2012.txt", dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Segments:
        return pp_segment(output_tensors)
