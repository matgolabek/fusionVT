import numpy as np
from sort import *
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def track(images_path: str, data_path:str):
    rgb_files = sorted(glob.glob('%s/rgb/*.*' % images_path))
    colors = np.random.rand(40, 3)
    rgb_imgs = []
    for img_i, rgb_path in enumerate(tqdm(rgb_files, desc="Loading data")):
        rgb_imgs.append(np.array(Image.open(rgb_path)))

    with open(data_path, "r") as f:
        bboxes_per_frame = json.load(f)

    tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.1)

    for i, bboxes in enumerate(tqdm(bboxes_per_frame, desc="Tracking")):
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(rgb_imgs[i])
        if not bboxes:
            bboxes = np.empty((0, 5))
        else:
            bboxes = np.array(bboxes)
        bboxes_with_idx = tracker.update(bboxes)
        
        for bbox_with_idx in bboxes_with_idx:
            x = bbox_with_idx[0]
            y = bbox_with_idx[1]
            w = bbox_with_idx[2] - x
            h = bbox_with_idx[3] - y
            idx = int(bbox_with_idx[4])
            color = colors[idx % 40]
            bbox = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x, y, s=str(idx), color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('track/%d.png' % i, bbox_inches='tight', pad_inches=0.0)
        plt.close('all')
    print('Done')


if __name__ == '__main__':
    track('PST_900_RGBT_Dataset/test', 'bboxes_per_frame.json')

