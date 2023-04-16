import pickle
from action_constants import Action
from pathlib import Path

labels_path = '../datasets_pose/merged/merged_pull_ups_jump_rope_push_ups.pkl'

with open(labels_path, 'rb') as file:
    data = pickle.load(file)

overall_frames = 0
overall_clips = 0
label_number = {
    0: 0,
    1: 0,
    2: 0,
    3: 0
}

for annotation in data['annotations']:
    overall_frames += annotation['total_frames']
    overall_clips += 1
    label_number[annotation['label']] += 1
 
print('Average frame number:', overall_frames / overall_clips)
for n in label_number.values():
    print(n)