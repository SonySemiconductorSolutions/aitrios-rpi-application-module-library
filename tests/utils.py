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
import shutil
import zipfile
import requests
from tqdm import tqdm


def get_imagenet_samples(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = [
        {
            'label_id': 1,
            'label': 'goldfish, Carassius auratus',
            'url': 'https://github.com/EliSchwartz/imagenet-sample-images/raw/master/n01443537_goldfish.JPEG?raw=true',
            'path': os.path.join(output_dir, "a_goldfish.jpg")
        },
        {
            'label_id': 605,
            'label': 'iPod',
            'url': 'https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n03584254_iPod.JPEG?raw=true',
            'path': os.path.join(output_dir, "b_ipod.jpg")
        },
        {
            'label_id': 985,
            'lable': 'daisy',
            'url': 'https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n11939491_daisy.JPEG?raw=true',
            'path': os.path.join(output_dir, "c_daisy.jpg")
        }
    ]

    for sample in data:
        if not os.path.exists(sample['path']):
            response = requests.get(sample['url'])
            with open(sample['path'], 'wb') as f:
                f.write(response.content)

    return data


def get_imagenet_dataset(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = get_imagenet_samples(output_dir)

    # move each image to a new directory with the class id as the directory name
    for sample in data:
        class_id = sample['label_id']
        class_dir = os.path.join(output_dir, str(class_id))
        os.makedirs(class_dir, exist_ok=True)
        shutil.move(sample['path'], class_dir + "/" + os.path.basename(sample['path']))

    return data


def get_coco_samples(output_dir):
    """
    mapping coco dataset 90 classes to model 80 classes
    see https://github.com/levan92/coco-classes-mapping/blob/master/coco_mapping_91to80.json
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = [
        {
            'image_id': 724,
            'classes': [2, 7, 11],
            'classes_90': [3, 8, 13],
            'classes_labels': ['car', 'truck', 'stop sign'],
            'url': 'https://farm4.staticflickr.com/3576/3383381911_9d6d5b63a1_z.jpg',
            'coco_explorer': 'https://cocodataset.org/#explore?id=724',
            'path': os.path.join(output_dir, "724.jpg")
        },
        {
            'image_id': 785,
            'classes': [0, 30],
            'classes_90': [1, 35],
            'classes_labels': ['person', 'skis'],
            'url': 'https://farm8.staticflickr.com/7015/6795644157_f019453ae7_z.jpg',
            'coco_explorer': 'https://cocodataset.org/#explore?id=785',
            'path': os.path.join(output_dir, "785.jpg")
        },
        {
            'image_id': 885,
            'classes': [0, 38],
            'classes_90': [1, 43],
            'classes_labels': ['person', 'tennis racket'],
            'url': 'https://farm4.staticflickr.com/3715/9639200419_ee41490b2a_z.jpg',
            'coco_explorer': 'https://cocodataset.org/#explore?id=885',
            'path': os.path.join(output_dir, "885.jpg")
        },
        {
            'image_id': 1490,
            'classes': [0, 37],
            'classes_90': [1, 42],
            'classes_labels': ['person', 'surfboard'],
            'url': 'https://farm9.staticflickr.com/8523/8624108829_4e9c77d2d5_z.jpg',
            'coco_explorer': 'https://cocodataset.org/#explore?id=1490',
            'path': os.path.join(output_dir, "1490.jpg")
        }
    ]

    for sample in data:
        if not os.path.exists(sample['path']):
            response = requests.get(sample['url'])
            with open(sample['path'], 'wb') as f:
                f.write(response.content)

    return data


def get_coco_keypoints_samples(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = [
        {
            'image_id': 7977,
            'url': 'https://farm3.staticflickr.com/2886/9921714023_4924210a15_z.jpg',
            'coco_explorer': 'https://cocodataset.org/#explore?id=7977',
            'path': os.path.join(output_dir, "7977.jpg")
        },
        {
            'image_id': 11699,
            'url': 'https://farm7.staticflickr.com/6010/5946492599_6248b4620e_z.jpg',
            'coco_explorer': 'https://cocodataset.org/#explore?id=11699',
            'path': os.path.join(output_dir, "11699.jpg")
        },
        {
            'image_id': 12639,
            'url': 'https://farm5.staticflickr.com/4102/4795012771_81c0b6b502_z.jpg',
            'coco_explorer': 'https://cocodataset.org/#explore?id=12639',
            'path': os.path.join(output_dir, "12639.jpg")
        },
    ]

    for sample in data:
        if not os.path.exists(sample['path']):
            response = requests.get(sample['url'])
            with open(sample['path'], 'wb') as f:
                f.write(response.content)

    return data


def get_tracking_video(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    info = {
        'url': 'https://github.com/ifzhang/ByteTrack/blob/main/videos/palace.mp4?raw=true',
        'path': os.path.join(output_dir, "palace.mp4")
    }

    if not os.path.exists(info['path']):
        response = requests.get(info['url'])
        with open(info['path'], 'wb') as f:
            f.write(response.content)

    return info


def get_coco_annotations(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = {
        'url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'path': os.path.join(output_dir, "annotations_trainval2017.zip")
    }

    if not os.path.exists(data['path']):
        response = requests.get(data['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(data['path'], 'wb') as f, tqdm(
            desc="Downloading coco annotations",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Extract the zip file
    with zipfile.ZipFile(data['path'], 'r') as zip_ref:
        if any(f.startswith('annotations/') for f in zip_ref.namelist()):
            # Check if already extracted
            needs_extraction = any(
                not os.path.exists(os.path.join(output_dir, f.removeprefix('annotations/')))
                for f in zip_ref.namelist()
                if f.startswith('annotations/') and not f.endswith('/') and f.removeprefix('annotations/')
            )

            # Extract contents of annotations/ folder directly to output_dir
            if needs_extraction:
                for f in zip_ref.infolist():
                    target_path = os.path.join(output_dir, f.filename.removeprefix('annotations/'))
                    if (
                        f.filename.startswith('annotations/') and
                        not f.is_dir() and
                        not os.path.exists(target_path)
                    ):
                        # Extract the file
                        with zip_ref.open(f) as source, open(target_path, 'wb') as target:
                            target.write(source.read())
        else:
            raise ValueError("Unexpected annotations content")

    return data


def get_voc_samples(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_dir = os.path.join(output_dir, "JPEGImages")
    mask_dir = os.path.join(output_dir, "SegmentationClass")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    data = [
        {
            'img_id': 2376,
            'img_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/002376.jpg',
            'class_mask_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/002376_class.png',
            # 'object_mask_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/002376_object.png',
            'img_path': os.path.join(output_dir, image_dir, "2376.jpg"),
            'mask_path': os.path.join(output_dir, mask_dir, "2376.png"),
        },
        {
            'img_id': 6585,
            'img_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/006585.jpg',
            'class_mask_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/006585_class.png',
            # 'object_mask_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/006585_object.png',
            'img_path': os.path.join(output_dir, image_dir, "6585.jpg"),
            'mask_path': os.path.join(output_dir, mask_dir, "6585.png"),
        },
        {
            'img_id': 7109,
            'img_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/007109.jpg',
            'class_mask_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/007109_class.png',
            # 'object_mask_url': 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/segexamples/images/007109_object.png',
            'img_path': os.path.join(output_dir, image_dir, "7109.jpg"),
            'mask_path': os.path.join(output_dir, mask_dir, "7109.png"),
        },
    ]

    for sample in data:
        if not os.path.exists(sample['img_path']):
            response = requests.get(sample['img_url'])
            with open(sample['img_path'], 'wb') as f:
                f.write(response.content)
        if not os.path.exists(sample['mask_path']):
            response = requests.get(sample['class_mask_url'])
            with open(sample['mask_path'], 'wb') as f:
                f.write(response.content)

    return data
