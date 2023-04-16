import json
from tqdm import tqdm
import shutil

with open('jump_rope_clean.json', 'r') as ann:
    data = json.load(ann)

id_to_name = {}

for im in data['images']:
    id_to_name[im['id']] = im['file_name']

for i, ann in enumerate(tqdm(data['annotations'])):
    image_name = id_to_name[ann['image_id']]
    image_path = f'../jump_rope/images/{image_name}'

    shutil.copy(image_path, f'jump_rope/images/{image_name}')