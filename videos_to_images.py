from pathlib import Path
import cv2
from tqdm import tqdm

def extract_images(video_dir, image_format='.png', extract_every=1):
    video_dir = Path(video_dir)
    image_dir = video_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in video_dir.iterdir():
        if video_path.is_dir():
            continue
        video_name = video_path.stem.replace(' ', '_')
        capture = cv2.VideoCapture(str(video_path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_number in tqdm(range(frame_count)):
            success, frame = capture.read()
            image_path = image_dir / f'{video_name}_{frame_number}{image_format}'
            if success:
                if frame_number % extract_every == 0:
                    cv2.imwrite(str(image_path), frame)
            else:
                break
            frame_number += 1
        capture.release()

video_dir = 'some_videos'
extract_every = 3
extract_images(video_dir, extract_every=extract_every)