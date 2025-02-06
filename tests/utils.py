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
import requests


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
        }
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