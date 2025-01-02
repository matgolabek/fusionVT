from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

from distutils.util import strtobool
import os
import sys
import time
import datetime
import argparse

import multiprocessing

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from brevitas.nn import QuantConv2d

Bool_arg = lambda x: bool(strtobool(x))

if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='data/PST900_RGBT_Dataset_MOD/test', help='path to dataset')
    parser.add_argument('--config_path', type=str, default='config/yolov3_quant.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='checkpoints/quant_499.weightd', help='path to weights file')
    #parser.add_argument("--weights_path", type=str, default="checkpoints/99.onnx", help="path to weights file")
    parser.add_argument('--class_path', type=str, default='data/pst900.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.9, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=Bool_arg, default=True, help='whether to use cuda if available')
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available() and opt.use_cuda

    os.makedirs('output', exist_ok=True)

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)

    file = open(opt.config_path,'r')
    first_line = file.readline().rstrip().lstrip().split('=')
    file.close()
    
    using_quantized_layers = False
    if len(first_line)>1: 
        if first_line[0]=='#quantization' and first_line[1]=='1':  
            using_quantized_layers = True

    if using_quantized_layers:
        pass
        # model.module_list[0][0]   = QuantConv2d(in_channels=4,out_channels=32,kernel_size=(3,3),stride=(1, 1),padding=(1,1),bias=False)
        # model.module_list[81][0]  = QuantConv2d(in_channels=1024,out_channels=27,kernel_size=(1,1),stride=(1, 1))
        # model.module_list[93][0]  = QuantConv2d(in_channels=512,out_channels=27,kernel_size=(1,1),stride=(1, 1))
        # model.module_list[105][0] = QuantConv2d(in_channels=256,out_channels=27,kernel_size=(1,1),stride=(1, 1))
    else:
        model.module_list[0][0]   = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.module_list[81][0]  = nn.Conv2d(1024, 27, kernel_size=(1, 1), stride=(1, 1))
        model.module_list[93][0]  = nn.Conv2d(512, 27, kernel_size=(1, 1), stride=(1, 1))
        model.module_list[105][0] = nn.Conv2d(256, 27, kernel_size=(1, 1), stride=(1, 1))

    model.load_weights_dict(opt.weights_path)

    if cuda:
        model.cuda()

    model.eval() # Set in evaluation mode

    dataloader = DataLoader(RGBTFolderX(opt.image_folder, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    classes = load_classes(opt.class_path) # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index

    print ('\nPerforming object detection:')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs, _) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)


        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print ('\nSaving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print ("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
        # Image height and width after padding is removed
        unpad_h = opt.img_size - pad_y
        unpad_w = opt.img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections.cpu():

                if int(cls_pred) == 4:
                    print("hello")

                print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                        edgecolor=color,
                                        facecolor='none')
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
        plt.close()