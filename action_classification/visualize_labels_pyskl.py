import pickle
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from action_constants import Action

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

label_path = '../mmaction2/data/skeleton/merged_pull_ups_jump_rope_push_ups.pkl'
with open(label_path, 'rb') as file:
    data = pickle.load(file)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 3

color = (255, 0, 0)

thickness = 2

label_map = {
    int(Action.PULL_UP): 'pull up',
    int(Action.PUSH_UP): 'push up',
    int(Action.JUMP_ROPE): 'jump_rope',
    int(Action.REST): 'rest'
}

for annotation in data['annotations']:
    height, width = annotation['img_shape']
    keypoints = annotation['keypoint'][0]
    label = label_map[annotation['label']]
    for idx, pose in enumerate(tqdm(keypoints)):
        pose = pose.reshape(pose.shape[0] * 2)
        blank_image = np.zeros((height, width, 3), np.uint8)
        plot_skeleton_kpts(blank_image, pose, steps=2)
        cv2.putText(blank_image, label, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        Path(f"vis/{annotation['frame_dir']}").mkdir(exist_ok=True, parents=True)
        cv2.imwrite(f"vis/{annotation['frame_dir']}/{idx}.png", blank_image)