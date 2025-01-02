import numpy as np
import glob
import cv2 as cv
from tqdm import tqdm
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import matplotlib
import json


matplotlib.use('TkAgg')


def make_bbox(folder_path: str, skip: int = 0, num_of_class: int = 4):
    rgb_files = sorted(glob.glob('%s/rgb/*.*' % folder_path))
    label_files = sorted(glob.glob('%s/labels/*.*' % folder_path))
    if skip > 0:
        rgb_files = rgb_files[skip:]
        label_files = label_files[skip:]

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    bbox_colors = random.sample(colors, num_of_class)
    classes = ['fire extinguisher', 'backpack', 'drill', 'human']

    data_to_log = []

    for img_i, (rgb_path, label_path) in enumerate(zip(tqdm(rgb_files), label_files)):
        rgb_img = np.array(Image.open(rgb_path))
        label_img = np.array(Image.open(label_path))
        img_h, img_w, _ = rgb_img.shape

        # Find contours in the segmentation image
        contours, _ = cv.findContours(label_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Iterate through the contours to get bounding boxes
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(rgb_img)
        bboxes = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            # Draw the rectangle on the image
            unique_pixels, counts = np.unique(label_img[y:y + h, x:x + w], return_counts=True)
            max = 0
            j_max = -1
            for j, pix in enumerate(unique_pixels):
                if pix == 0 or pix == 255:
                    continue
                if counts[j] > max:
                    max = counts[j]
                    j_max = j
            object_class = 256
            if j_max != -1:
                object_class = unique_pixels[j_max]
            color = bbox_colors[object_class - 1]

            bbox = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            bboxes.append([x, y, x + w, y + h, 1.0])
            # Add label
            plt.text(x, y, s=classes[object_class - 1], color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('output/%d.png' % (skip + img_i), bbox_inches='tight', pad_inches=0.0)
        plt.close('all')
        data_to_log.append(bboxes)
    with open('bboxes_per_frame.json', 'w') as f:
        json.dump(data_to_log, f)


if __name__ == '__main__':
    make_bbox('PST_900_RGBT_Dataset/test')
