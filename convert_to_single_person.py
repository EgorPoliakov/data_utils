import json
from pathlib import Path
import cv2
from tqdm import tqdm
import numpy as np

with open('dancing_datasets1_single_person.json', 'r') as ann:
    data = json.load(ann)

id_to_name = {}
cropped_data = {
    'annotations': [],
    'images': [],
    'info': {},
    "licenses": [
        {
            "id": 0,
            "name": "Lices",
            "url": "url"
        }
    ],
    "categories": [
        {
            "supercategory": "person",
            "name": "person",
            "skeleton": [
                [
                    16,
                    14
                ],
                [
                    14,
                    12
                ],
                [
                    17,
                    15
                ],
                [
                    15,
                    13
                ],
                [
                    12,
                    13
                ],
                [
                    6,
                    12
                ],
                [
                    7,
                    13
                ],
                [
                    6,
                    7
                ],
                [
                    6,
                    8
                ],
                [
                    7,
                    9
                ],
                [
                    8,
                    10
                ],
                [
                    9,
                    11
                ],
                [
                    2,
                    3
                ],
                [
                    1,
                    2
                ],
                [
                    1,
                    3
                ],
                [
                    2,
                    4
                ],
                [
                    3,
                    5
                ],
                [
                    4,
                    6
                ],
                [
                    5,
                    7
                ]
            ],
            "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle"
            ],
            "id": 1
        }
    ]    
}

for im in data['images']:
    id_to_name[im['id']] = im['file_name']

for i, ann in enumerate(tqdm(data['annotations'])):
    image_name = id_to_name[ann['image_id']]
    image_path = f'../aist_images/{image_name}'

    image = cv2.imread(image_path)
    bbox = ann['bbox']
    
    keypoints = ann['keypoints']
    keypoints = np.array(keypoints)
    
        
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    expand_x = int(x * 0.2)
    expand_y = int(y * 0.05)

    x_min = x - expand_x
    x_max = x + w + expand_x
    y_min = y - expand_y
    y_max = y + h + expand_y

    keypoints[0::3] -= x_min
    keypoints[1::3] -= y_min
    image = image[y_min:y_max, x_min:x_max]
    # for x, y in zip(keypoints[0::3], keypoints[1::3]):
    #     cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
    annotation = {
        'keypoints': list(keypoints),
        'bbox': [0.0, 0.0, 0.0, 0.0],
        'image_id': i,
        'id': i,
        'category_id': 1,
        'num_keypoints': 17,
        'area': (y_max - y_min) * (x_max - x_min)
    }

    image_description = {
        'file_name': f'{image_name.replace(".png", "")}_{i}.png',
        "license": 0,
        "height": image.shape[0],
        "width": image.shape[1],
        "id": i
    }

    cropped_data['annotations'].append(annotation)
    cropped_data['images'].append(image_description)
    cv2.imwrite(f'../dancing_dataset1_single_person_expand/images/train2017/{image_name.replace(".png", "")}_{i}.png', image)
    
with open('../dancing_dataset1_single_person_expand/dancing_dataset1_single_person_expand.json', 'w') as f:
    json.dump(cropped_data, f)