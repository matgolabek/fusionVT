#!/bin/bash

python3 ./train.py \
    --epochs 200 \
    --image_folder "data/PST900_RGBT_Dataset" \
    --batch_size 4 \
    --model_config_path "config/yolov3.cfg" \
    --data_config_path "config/pst900.data" \
    --weights_path "weights/float_0e.weightd" \
    --class_path "data/pst900.names" \
    --conf_thres 0.8 \
    --nms_thres 0.4 \
    --n_cpu 0 \
    --img_size 416 \
    --checkpoint_interval 1 \
    --checkpoint_dir "checkpoints" \
    --use_cuda True \
    --freeze False \
    --modw True \
    --ema True \
    --mAP True