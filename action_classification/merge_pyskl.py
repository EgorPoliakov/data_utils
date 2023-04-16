import pickle
from pathlib import Path

def merge_pkls(labels_dir, output_path):
    merged_labels = {
        'annotations': [],
        'split': {
            'train': [],
            'val': []
        }
    }

    label_names_to_merge = []

    for label_path in Path(labels_dir).iterdir():
        with open(label_path, 'rb') as file:
            data = pickle.load(file)
            label_names_to_merge.append(label_path.stem)
            merged_labels['annotations'].extend(data['annotations'])
            merged_labels['split']['train'].extend(data['split']['train'])
            merged_labels['split']['val'].extend(data['split']['val'])

    output_path = output_path.replace('.pkl', '_' + '_'.join(label_names_to_merge) + '.pkl')

    with open(output_path, 'wb') as file:
        pickle.dump(merged_labels, file)

    print('Number of merged dataset annotations:', len(merged_labels['annotations']))
merge_pkls('../datasets_pose/to_merge', '../datasets_pose/merged/merged.pkl')