from __future__ import division

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from copy import deepcopy

from PIL import Image

from utils.parse_config import *
from utils.utils import build_targets, is_parallel
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from pytorch_nndct.nn import QuantStub, DeQuantStub

# class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
#     """
#     Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
#     specified by each layer.
#     """
#     scaling_min_val = 2e-16
#     bit_width = None


# class CommonIntWeightPerChannelQuant(CommonIntWeightPerTensorQuant):
#     """
#     Common per-channel weight quantizer with bit-width set to None so that it's forced to be
#     specified by each layer.
#     """
#     scaling_per_output_channel = True


# class CommonIntActQuant(Int8ActPerTensorFloat):
#     """
#     Common signed act quantizer with bit-width set to None so that it's forced to be specified by
#     each layer.
#     """
#     scaling_min_val = 2e-16
#     bit_width = None
#     restrict_scaling_type = RestrictValueType.LOG_FP


# class CommonUintActQuant(Uint8ActPerTensorFloat):
#     """
#     Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
#     each layer.
#     """
#     scaling_min_val = 2e-16
#     bit_width = None
#     restrict_scaling_type = RestrictValueType.LOG_FP

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "quant_convolutional":
            bn = int(module_def["quant_batch_normalize"])
            rqt = bool(module_def["return_quant_tensor"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "quant_conv_%d" % i,
                qnn.QuantConv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=8
                    #bias_quant=Int8Bias,
                    #input_quant=Int8ActPerTensorFloat,
                    #output_quant=Int8ActPerTensorFloat,
                    #return_quant_tensor=rqt
                ),
            )
            if bn:
                # modules.add_module("quant_batch_norm_%d" % i, quant_bn.BatchNorm2dToQuantScaleBias(filters,
                #     weight_quant=Int8WeightPerTensorFloat,
                #     bias_quant=Int8Bias,
                #     input_quant=Int8ActPerTensorFloat,
                #     output_quant=Int8ActPerTensorFloat,
                #     return_quant_tensor=True))
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))

            if module_def["activation"] == "quant_relu":
                modules.add_module("quant_relu_%d" % i, qnn.QuantReLU(act_quant=CommonUintActQuant,
                                                                      bit_width=8,
                                                                      #return_quant_tensor=rqt,
                                                                      scaling_per_channel = True
                                                                      ))
            # else:
            #     modules.add_module("quant_linear_%d" % i, QuantIdentity(return_quant_tensor=False))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "quant_upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            #upsample = QuantUpsample(scale_factor=int(module_def["stride"]),return_quant_tensor=False)
            modules.add_module("reg_upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())
            # modules.add_module("route_%d" % i, QuantIdentity(return_quant_tensor=False))

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())
            # modules.add_module("shortcut_%d" % i, QuantIdentity(return_quant_tensor=False))

        elif module_def["type"] == "yolo":
            print("yolo num:", i)
            print("Mask:", module_def["mask"])
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            print("Anchors:", module_def["anchors"])
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            print("Anchors:", anchors)
            num_classes = int(module_def["classes"])
            print("Classes:", module_def["classes"])
            img_height = int(hyperparams["height"])
            print("Height:", img_height)
            # Define detection layer
            print(i, anchors, num_classes, img_height)
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.anchors_tensor = torch.tensor(self.anchors)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss
        print("Init done")

    def forward(self, x, targets=None):
        # if isinstance(x, qnn.QuantTensor):
        #     x = x.tensor
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        Tensor = torch.cuda.Tensor if x.is_cuda else torch.Tensor
        DoubleTensor = torch.cuda.DoubleTensor if x.is_cuda else torch.DoubleTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
#          scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        scaled_anchors = torch.div(self.anchors_tensor, stride)
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals) if nProposals !=0 else float(0)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
#             tcls = Variable(tcls.type(DoubleTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        #self.quant_input = qnn.QuantIdentity(bit_width=8,return_quant_tensor=True,output_quant=Int8ActPerTensorFloat)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()
        print("new init 10")

        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_output_name = None

    def forward(self, x, targets=None):
        # if self.onnx_session is not None:
        #     if isinstance(x, torch.Tensor):
        #         x = x.detach().cpu().numpy()  # Convert to NumPy array if input is a PyTorch tensor
        #     outputs = self.onnx_session.run([self.onnx_output_name], {self.onnx_input_name: x})
        #     return torch.tensor(outputs[0])  # Convert ONNX output back to PyTorch tensor

        #x = self.quant_input(x)
        # x = self.quant(x) # QUANTSTUB TRY
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample","quant_upsample", "maxpool","quant_convolutional"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
#                 x = self.dequant(x) # QUANTSTUB TRY
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    pass
                    # x = self.dequant(x) # QUANTSTUB TRY
                    # print("Before YOLO", type(x), x.shape, x)
#                     x = module(x)
#                     print("After YOLO", type(x), x.shape, x[0, 0, :])
                output.append(x)
#                 outputs.append((x, module))
                outputs.append(x)
#                 x = self.quant(x) # QUANTSTUB TRY
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
#         return sum(output) if is_training else torch.cat(output, 1)
        return outputs
#         return torch.cat(output, 1)

    def process_yolo_layers(self, darknet_outputs):
        output = []

        for x, module in darknet_outputs:
            x = module(x)
            output.append(x)

        return torch.cat(output, 1)

    def load_weights_pytorch(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights !!!!!! cahnged from np.float32 to np.int8
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
            
            elif module_def["type"] == "quant_convolutional":
                conv_layer = module[0]
                if module_def["quant_batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def load_weights(self, weights_path):
        self.onnx_session = onnxruntime.InferenceSession(weights_path)
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_session.get_outputs()[0].name
        print(f"Weights successfully loaded from {weights_path}")


    """
        @:param path    - path of the new weights file
        @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
    """

    def load_weights_dict(self, weights_path):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
#         self.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))

    def save_weights_dict(self, path, cutoff=-1):
        torch.save(self.state_dict(), path) 

    def save_weights_pytorch(self, path, cutoff=-1):

        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)
            
            elif module_def["type"] == "quant_convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["quant_batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

    def save_weights(self, path, cutoff=-1):
        # Adjust your input shape as needed for your model
        input_shape = (1, 4, self.img_size, self.img_size)  # Example input shape with 4 channels

        self.eval()
        self.cpu()
        # Export the model to an ONNX file
        start = time.time()
        export_onnx_qcdq(
            self,
            args = torch.rand(input_shape),
            use_external_data_format=True,
            #input_shape=input_shape,
            export_path=path,
            opset_version=13
        )
        end = time.time()
        print(f"Export time : {end-start}")
        if torch.cuda.is_available():
            self.cuda()
        self.train()
        print(f"Quantized weights successfully saved to {path}")


class ModelEMA:
    def __init__(self, model, decay = 0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval() #FP32 EMA
        self.device = next(model.parameters()).device
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state.dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v = v.to(self.device)
                    v *= d
                    v += (1. - d) * msd[k].detach()
    
    # def update_attr(self, model, include=(), exclude=('process_group','reducer')):
    #     copy_attr(self.ema, model, include, exclude)

