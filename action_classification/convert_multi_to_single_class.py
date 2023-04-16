import pickle
from action_constants import Action

labels_path = 'extracted_0_1_2_3_27_31_32.pkl'
single_class = int(Action.REST)

with open(labels_path, 'rb') as file:
    data = pickle.load(file)

for idx, annotation in enumerate(data['annotations']):
    data['annotations'][idx]['label'] = single_class

with open(f'single_class_{single_class}.pkl', 'wb') as file:
    pickle.dump(data, file)