from __future__ import division
from test import test

from distutils.util import strtobool
from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from copy import deepcopy
import os
from re import compile as compile_re
from re import Pattern
from glob import glob as glob_glob
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from brevitas.nn import QuantConv2d
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloat

Bool_arg = lambda x: bool(strtobool(x))

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/PST900_RGBT_Dataset", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
# parser.add_argument("--model_config_path", type=str, default="config/yolov3_quant.cfg", help="path to model config file")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="path to model config file")
parser.add_argument("--data_config_path", type=str, default="config/pst900.data", help="path to data config file")
# parser.add_argument("--weights_path", type=str, default="weights/199.weightd", help="path to weights file")
parser.add_argument("--weights_path", type=str, default="weights/float_0e.weightd", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/pst900.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=  416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved")
parser.add_argument("--use_cuda", type=Bool_arg, default=True, help="whether to use cuda if available")
parser.add_argument("--freeze", type=Bool_arg, default=False, help="whether to freeze most of the tensors, [0,81,93,105] stay unfrozen")
parser.add_argument("--modw", type=Bool_arg, default=True, help="check if you're using non standard yolov3 layers")
parser.add_argument("--ema", type=Bool_arg, default=False, help="enable for EMA")
parser.add_argument("--mAP", type=Bool_arg, default=False, help="calculate mAP every epoch")
opt = parser.parse_args()
print(opt)

kwarg_for_test = {'batch_size': opt.batch_size,
                  'model_config_path': opt.model_config_path,
                  'data_config_path': opt.data_config_path,
                  'class_path': opt.class_path,
                  'n_cpu': opt.n_cpu,
                  'img_size': opt.img_size,
                  'use_cuda': opt.use_cuda}

cuda = torch.cuda.is_available() and opt.use_cuda

os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

#Set log path
log_path = "./logs/log_001.txt"
if os.path.isfile(log_path):
    last_log_entry = sorted(glob_glob("./logs/log_*.txt"))[-1]
    pattern = compile_re("(log_)(\d+)(\.txt)")
    prefix,number,suffix  = pattern.search(last_log_entry).groups()
    lenght_of_pad_string = len(number)
    number = str(int(number)+1).rjust(lenght_of_pad_string,'0')
    log_path = "./logs/"+prefix+number+suffix
    
classes = load_classes(opt.class_path)

# Get data configuration
data_config = parse_data_config(opt.data_config_path)

train_path = data_config["train"]
test_path = data_config["valid"]

# Get hyper parameters
hyperparams = parse_model_config(opt.model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(opt.model_config_path)
#model.apply(weights_init_normal)

file = open(opt.model_config_path,'r')
first_line = file.readline().rstrip().lstrip().split('=')
file.close()

using_quantized_layers = False
if len(first_line)>1: 
    if first_line[0].rstrip().lstrip()=='#quantization' and first_line[1].rstrip().lstrip()=='1':  
        using_quantized_layers = True

if not opt.modw: model.load_weights_pytorch(opt.weights_path)

# if using_quantized_layers:
#     pass
#     # model.module_list[0][0]   = QuantConv2d(in_channels=4,out_channels=32,kernel_size=(3,3),stride=(1, 1),padding=(1,1),
#     #                                         bias=False,)
#     # model.module_list[81][0]  = QuantConv2d(in_channels=1024,out_channels=27,kernel_size=(1,1),stride=(1, 1))
#     # model.module_list[93][0]  = QuantConv2d(in_channels=512,out_channels=27,kernel_size=(1,1),stride=(1, 1))
#     # model.module_list[105][0] = QuantConv2d(in_channels=256,out_channels=27,kernel_size=(1,1),stride=(1, 1))
# else:
#     pass
#     # model.module_list[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#     # model.module_list[81][0] = nn.Conv2d(1024, 27, kernel_size=(1, 1), stride=(1, 1))
#     # model.module_list[93][0] = nn.Conv2d(512, 27, kernel_size=(1, 1), stride=(1, 1))
#     # model.module_list[105][0] = nn.Conv2d(256, 27, kernel_size=(1, 1), stride=(1, 1))


if opt.modw: model.load_weights_dict(opt.weights_path)
# END Initiate model
 
#if freeze param is True
#Only train on specified layers
if opt.freeze:
    to_train = [0, 81, 93, 105]
    for k, v in model.named_parameters():
        v.requires_grad = False
        if int(k.split('.')[1]) in to_train:
            v.requires_grad = True

if cuda:
    model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    # RGBTFolderX(train_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
    RGBTFolderP(train_path, train=True), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

test_dataloader = torch.utils.data.DataLoader(RGBTFolderP(test_path), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

if opt.ema: ema = ModelEMA(model)

for epoch in range(opt.epochs):
    info_to_log = ''
    for batch_i, batch in enumerate(dataloader):
        (_, imgs, targets) = batch
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        
        optimizer.zero_grad()
        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()
        if opt.ema: ema.update(model)
        
        log_info = (f"[Epoch {epoch}/{opt.epochs}, Batch {batch_i}/{len(dataloader)}]\n"
                       f"[Losses: x {model.losses['x']}, y {model.losses['y']}, w {model.losses['w']}, h {model.losses['h']}, "
                       f"conf {model.losses['conf']}, cls {model.losses['cls']}, total {loss.item()}, " 
                       f"recall: {model.losses['recall']:.5f}, precision: {model.losses['precision']:.5f}]\n")
        
        info_to_log += log_info
        print(log_info)
    
    if opt.mAP:
        if opt.ema:
            test_info = test(use_train_model=ema.ema, use_loader=test_dataloader, use_parser=False, suppress_return=False, **kwarg_for_test)
        else:
            test_info = test(use_train_model=deepcopy(model), use_loader=test_dataloader, use_parser=False, suppress_return=False, **kwarg_for_test)
     
        log_info = (f"[C0_AP: {test_info[0]}, C1_AP: {test_info[1]}, C2_AP: {test_info[2]}, C3_AP: {test_info[3]}, "
                            f"mAP: {test_info['mAP']}]\n")
        
        info_to_log += log_info
        print(log_info)

    with open(log_path, "a") as f:
        f.write(info_to_log)

    if epoch % opt.checkpoint_interval == 0:
        if using_quantized_layers:
            model.save_weights_dict("%s/quant_%d.weightd" % (opt.checkpoint_dir, epoch))
        else:
            model.save_weights_dict("%s/float_%d.weightd" % (opt.checkpoint_dir, epoch))
