import cv2
from tqdm import tqdm
import json
import matplotlib

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def convert_predictions(json_path, image_dir, output_path, image_format='.jpg'):
    labels = read_json(json_path)
    data = {}
    data['annotations'] = labels
    
    images = []
    overall_annotations = []
    image_name_to_annotation = {}
    for annotation in data['annotations']:
        if annotation['image_id'] not in image_name_to_annotation:
            image_name_to_annotation[annotation['image_id']] = []
        image_name_to_annotation[annotation['image_id']].append(annotation)
    
    for idx, (image_name, annotations) in enumerate(tqdm(image_name_to_annotation.items())):
        person_annotations = []
        image_path = f'{image_dir}/{image_name}{image_format}'
        for i, annotation in enumerate(annotations):
            annotation['image_id'] = idx
            annotation['id'] = idx
            if annotation['category_id'] == 1: #only person class
                person_annotations.append(annotation)
        overall_annotations.extend(person_annotations)
        image = cv2.imread(str(image_path))
        coco_image = {
            "license": 0,
            "file_name": image_path.split('/')[-1],
            "height": image.shape[0],
            "width": image.shape[1],
            "id": idx
        }
        images.append(coco_image)
        
    data['info'] = {}
    data['licenses'] = [
        {
            "id": 0,
            "name": "Lices",
            "url": "url",
        }]
    data['categories'] = [{"supercategory": "person", "name": "person", "skeleton": [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]], "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"], "id": 1}]
    data['annotations'] = overall_annotations
    data['images'] = images
    print(len(overall_annotations))
    with open(output_path, 'w') as f:
        json.dump(data, f)

convert_predictions(
    '../edgeai-yolov5/runs/test/exp11/pose-large-960_predictions.json', 
    '../jump_rope_images_every3', 
    'jump_rope.json', 
    image_format='.png'
)