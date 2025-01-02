from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from distutils.util import strtobool
import os
import sys
import time
import datetime
import argparse
import tqdm
from collections import namedtuple
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

# from brevitas.nn import QuantConv2d
# from brevitas.quant import Uint8ActPerTensorFloat

Bool_arg = lambda x: bool(strtobool(x))

def test(use_train_model = None, use_loader = None, use_parser = True, suppress_return = True, **kwargs):
    if use_parser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
        parser.add_argument("--model_config_path", type=str, default="config/yolov3_quant.cfg", help="path to model config file")
        parser.add_argument("--data_config_path", type=str, default="config/pst900.data", help="path to data config file")
        parser.add_argument("--weights_path", type=str, default="checkpoints/quant_499.weightd", help="path to weights file")
        #parser.add_argument("--weights_path", type=str, default="checkpoints/8.onnx", help="path to weights file")
        parser.add_argument("--class_path", type=str, default="data/pst900.names", help="path to class label file")
        parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
        parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
        parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
        parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        parser.add_argument("--use_cuda", type=Bool_arg, default=True, help="whether to use cuda if available")
        opt = parser.parse_args()
        print(opt)
    else:
        Argz = namedtuple("Argz",['batch_size',
                                'model_config_path',
                                'data_config_path',
                                'weights_path',
                                'class_path',
                                'iou_thres',
                                'conf_thres',
                                'nms_thres',
                                'n_cpu',
                                'img_size',
                                'use_cuda'])
        print(kwargs)
        opt = Argz(**kwargs)
        print(opt)
        
    cuda = torch.cuda.is_available() and opt.use_cuda

    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    test_path = data_config["valid"]
    num_classes = int(data_config["classes"])

    # Initiate model
    if not use_train_model:
        model = Darknet(opt.model_config_path)

        file = open(opt.model_config_path,'r')
        first_line = file.readline().rstrip().lstrip().split('=')
        file.close()

        using_quantized_layers = False
        if len(first_line)>1: 
            if first_line[0]=='#quantization' and first_line[1]=='1':  
                using_quantized_layers = True

        if using_quantized_layers:
            pass
            # model.module_list[0][0]   = QuantConv2d(in_channels=4,out_channels=32,kernel_size=(3,3),stride=(1, 1),padding=(1,1),
            #                                         bias=False)
            # model.module_list[81][0]  = QuantConv2d(in_channels=1024,out_channels=27,kernel_size=(1,1),stride=(1, 1)
            #                                         )
            # model.module_list[93][0]  = QuantConv2d(in_channels=512,out_channels=27,kernel_size=(1,1),stride=(1, 1)
            #                                         )
            # model.module_list[105][0] = QuantConv2d(in_channels=256,out_channels=27,kernel_size=(1,1),stride=(1, 1)
                                                    # )
        else:
            pass
            # model.module_list[0][0]   = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # model.module_list[81][0]  = nn.Conv2d(1024, 27, kernel_size=(1, 1), stride=(1, 1))
            # model.module_list[93][0]  = nn.Conv2d(512, 27, kernel_size=(1, 1), stride=(1, 1))
            # model.module_list[105][0] = nn.Conv2d(256, 27, kernel_size=(1, 1), stride=(1, 1))

        model.load_weights_dict(opt.weights_path)

    else:
        model = use_train_model
    
    if cuda:
        model = model.cuda()

    model.eval()

    # Get dataloader
    if not use_loader:
        dataset = RGBTFolderP(test_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    else:
        dataloader = use_loader
        
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    print("Compute mAP...")

    all_detections = []
    all_annotations = []

#     a = model.module_list
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)


        # img = imgs[0].cpu().numpy()
        # img = img[:3] * 255
        # img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

        # target = targets[0].cpu().numpy()

        # print(type(img), img.shape, img.dtype)
        # for t in target:
        #     cls, x, y, w, h = t

        #     if x != 0 and y != 0 and w != 0 and h != 0:
        #         cx = int(x * 416)
        #         cy = int(y * 416)
        #         w = int(w * 416)
        #         h = int(h * 416)

        #         x = cx - w // 2
        #         y = cy - h // 2

        #         img = np.ascontiguousarray(img)

        #         img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #         cv.imshow("img", img)
        #         cv.waitKey(0)


        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, 4, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)
            pass

        for output, annotations in zip(outputs, targets):

            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                # Get predicted boxes, confidence scores and labels
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                # Order by confidence
                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]
                
                print("Batch:", batch_i)
                print("pred_boxes:", pred_boxes)
                print("scores:", scores)
                print("pred_labels:", pred_labels)

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotations[:, -1] > 0):

                annotation_labels = annotations[annotations[:, -1] > 0, 0].cpu().numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:].cpu()

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= opt.img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in tqdm.tqdm(range(len(all_annotations)), desc=f"Computing AP for class '{label}'"):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= opt.iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    logger = {}
    print("Average Precisions:")
    for c, ap in average_precisions.items():
        print(f"+ Class '{c}' - AP: {ap}")
        logger[c] = ap

    mAP = np.mean(list(average_precisions.values()))
    logger["mAP"] = mAP
    print(f"mAP: {mAP}")

    if not suppress_return:
        model.train()
    return None if suppress_return else logger

if __name__ == "__main__":
    print(test(suppress_return=False))