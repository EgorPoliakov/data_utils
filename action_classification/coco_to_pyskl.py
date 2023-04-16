import pickle
import json
import numpy as np
from action_constants import Action

# pyskl_data = {
#     'split': {},
#     'annotations': [
#         {
#             'frame_dir': 'Video identifier',
#             'total_frames': 0,
#             'img_shape': (100, 100), #Height x Width
#             'original_shape': (100, 100), #Height x Width
#             'label': 0, #Action label
#             'keypoint': np.array([]), #Keypoints (1, n_frames, n_keypoints, 2)
#             'keypoint_score': np.array([]), #Confidence scores (1, n_frames, n_keypoints)

#         }
#     ]
# }

class CocoToPysklConverter:
    def __init__(self, coco_annotations_path, action_label, n_keypoints=17, split='train', frame_number_separator='_'):
        self.coco_annotations_path = coco_annotations_path
        self.action_label = int(action_label)
        self.n_keypoints = n_keypoints
        self.frame_number_separator = frame_number_separator
        self.coco_data = self.read_json()
        self.split = split
        self.pyskl_data = {
            'annotations': [],
            'split': {
                'train': [],
                'val': []
            }
        }

    def read_json(self):
        with open(self.coco_annotations_path, 'r') as f:
            data = json.load(f)
        return data

    def convert_coco_to_pyskl(self):
        id_to_image_name, id_to_image_description = self.map_image_id_to_name_and_description()
        video_name_to_annotations = self.map_video_name_to_annotations(id_to_image_name)
        pyskl_annotations = []

        for video_name, annotations in video_name_to_annotations.items():
            annotations.sort(key=lambda x: int(x['file_name'].split(self.frame_number_separator)[-1].split('.')[0]))
            
        for video_name, annotations in video_name_to_annotations.items():
            pyskl_annotation = self.build_pyskl_annotation(video_name, annotations, id_to_image_description)
            pyskl_annotations.append(pyskl_annotation)

        self.pyskl_data['annotations'] = pyskl_annotations
        self.pyskl_data['split'][self.split] = [ann['frame_dir'] for ann in pyskl_annotations]

    def save_pyskl(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self.pyskl_data, file)

    def map_image_id_to_name_and_description(self):
        id_to_image_name = {}
        id_to_image_description = {}

        for image_description in self.coco_data['images']:
            image_id = image_description['id']
            image_name = image_description['file_name']
            id_to_image_name[image_id] = image_name
            id_to_image_description[image_id] = image_description
        return id_to_image_name, id_to_image_description

    def map_video_name_to_annotations(self, id_to_image_name):
        video_name_to_annotations = {}
        for annotation in self.coco_data['annotations']:
            image_name = id_to_image_name[annotation['image_id']]
            video_name = self.frame_number_separator.join(image_name.split(self.frame_number_separator)[:-1])
            annotation['file_name'] = image_name
            if not video_name_to_annotations.get(video_name):
                video_name_to_annotations[video_name] = []
            video_name_to_annotations[video_name].append(annotation)
        return video_name_to_annotations

    def build_pyskl_annotation(self, video_name, annotations, id_to_image_description):
        pyskl_annotation = {}
        pyskl_annotation['frame_dir'] = video_name
        pyskl_annotation['total_frames'] = len(annotations)
        frame_height = id_to_image_description[annotations[0]['image_id']]['height']
        frame_width = id_to_image_description[annotations[0]['image_id']]['width']
        pyskl_annotation['img_shape'] = (frame_height, frame_width)
        pyskl_annotation['original_shape'] = (frame_height, frame_width)
        pyskl_annotation['label'] = self.action_label

        keypoint_coordinates, keypoint_confidences = self.convert_keypoints(annotations)

        pyskl_annotation['keypoint'] = keypoint_coordinates
        pyskl_annotation['keypoint_score'] = keypoint_confidences
        return pyskl_annotation

    def convert_keypoints(self, annotations):
        keypoints = np.array([x['keypoints'] for x in annotations])
        confidence_idx = np.zeros(keypoints.shape[1], dtype=bool)
        confidence_idx[2::3] = 1
        keypoint_coordinates = keypoints[:, ~confidence_idx]
        keypoint_confidences = keypoints[:, confidence_idx]
        
        keypoint_coordinates = keypoint_coordinates.reshape((
            1, 
            len(annotations), 
            self.n_keypoints, 
            2
        ))
        keypoint_confidences = keypoint_confidences.reshape((
            1,
            len(annotations),
            self.n_keypoints
        ))
        return keypoint_coordinates, keypoint_confidences

    


converter = CocoToPysklConverter('pull_ups.json', Action.PULL_UP)
converter.convert_coco_to_pyskl()
converter.save_pyskl('pull_ups.pkl')
with open('pull_ups.pkl', 'rb') as f:
    data = pickle.load(f)
annotation = data['annotations'][0]
print(annotation['total_frames'])
print(annotation['keypoint'].shape)