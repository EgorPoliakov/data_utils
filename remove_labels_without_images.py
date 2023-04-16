import json
from pathlib import Path

def remove_labels_without_images(json_path, images_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_dir = Path(images_path)
    image_path_set = set()

    print(f'Before cleaning: {len(data["annotations"])}')
    for image_path in image_dir.iterdir():
        image_path_set.add(image_path.name)

    image_id_set = set()
    new_images = []
    for image in data['images']:
        if image['file_name'] in image_path_set:
            new_images.append(image)
            image_id_set.add(image['id'])

    data['images'] = new_images

    new_annotations = []
    for ann in data['annotations']:
        if ann['image_id'] in image_id_set:
            new_annotations.append(ann)

    data['annotations'] = new_annotations
    print(f'After cleaning: {len(data["annotations"])}')

    output_json_path = f'{json_path.replace(".json", "_clean.json")}'
    with open(output_json_path, 'w') as f:
        json.dump(data, f)

remove_labels_without_images('jump_rope.json', 'jump_rope/predictions')