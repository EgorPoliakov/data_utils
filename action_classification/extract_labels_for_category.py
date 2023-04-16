import pickle

labels_path = '../mmaction2/data/skeleton/ntu60_2d.pkl'
labels_to_extract = [0, 1, 2, 3, 27, 31, 32]

with open(labels_path, 'rb') as file:
    data = pickle.load(file)

frame_extract_every = 3
clip_extract_every = 400
frame_count_train = 0
frame_count_val = 0

idx = 0
extracted_labels = {
    'annotations': [],
    'split': {
        'train': [],
        'val': []
    }
}

extracted_annotations = []
extracted_split_train = []
extracted_split_val = []
train_frame_names = set(data['split']['xsub_train'])
val_frame_names = set(data['split']['xsub_val'])

for annotation in data['annotations']:
    if annotation['label'] in labels_to_extract:
        idx += 1
        if idx % clip_extract_every == 0 and (annotation['frame_dir'] in train_frame_names or annotation['frame_dir'] in val_frame_names):
            annotation['keypoint'] = annotation['keypoint'][:, ::frame_extract_every, :, :]
            annotation['keypoint_score'] = annotation['keypoint_score'][:, ::frame_extract_every, :]
            annotation['total_frames'] = annotation['keypoint'].shape[1]
            extracted_annotations.append(annotation)
            if annotation['frame_dir'] in train_frame_names:
                frame_count_train += annotation['total_frames']
                extracted_split_train.append(annotation['frame_dir'])

            if annotation['frame_dir'] in val_frame_names:
                frame_count_val += annotation['total_frames']
                extracted_split_val.append(annotation['frame_dir'])

extracted_labels['annotations'] = extracted_annotations
extracted_labels['split']['train'] = extracted_split_train
extracted_labels['split']['val'] = extracted_split_val

output_file_name = f"extracted_{'_'.join([str(x) for x in labels_to_extract])}.pkl"
with open(output_file_name, 'wb') as file:
    pickle.dump(extracted_labels, file)

print('Number of extracted clips:', len(extracted_annotations))
print('Number of extracted train frames:', int(frame_count_train / frame_extract_every))
print('Number of extracted val frames:', int(frame_count_val / frame_extract_every))