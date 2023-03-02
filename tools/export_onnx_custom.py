#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module
import cv2
import time
import numpy as np


class CustomOutputModel(nn.Module):
    def __init__(self, model, class_num, conf_thre = 0.4, nms_thre=0.45) -> None:
        super(CustomOutputModel, self).__init__()
        self.model = model
        self.class_num = class_num
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre

    def forward(self, x):
        outputs_clone = self.model(x)

        bbox_out = outputs_clone[:, :, :4]
        bbox_out[:, :, 0] = bbox_out[:, :, 0] - bbox_out[:, :, 2] / 2
        bbox_out[:, :, 1] = bbox_out[:, :, 1] - bbox_out[:, :, 3] / 2
        bbox_out[:, :, 2] = bbox_out[:, :, 0] + bbox_out[:, :, 2] / 2
        bbox_out[:, :, 3] = bbox_out[:, :, 1] + bbox_out[:, :, 3] / 2

        objectness = torch.unsqueeze(outputs_clone[:, :, 4], 2)
        class_scores = outputs_clone[:, :, 5:]

        class_out = torch.mul(objectness , class_scores)

        outputs = torch.stack((bbox_out, class_out), 2)

        return outputs




def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=12, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--decode_in_inference",
        action="store_true",
        help="decode in inference or not"
    )

    return parser



def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference
    print(args.decode_in_inference)

    img_path = "/home/robotis-workstation3/ai/YOLOX-ROBOTIS/test/2023_2_7_11_6_21.jpg"
    img = cv2.imread(img_path)

    # # img, ratio = preproc(img, exp.test_size)
    # # img = torch.from_numpy(img).unsqueeze(0)
    # # img = torch.unsqueeze(img, 0)
    # img = img.float()

    # ratio = (exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])

    # custom_output_model = CustomOutputModel(model, 1)
    # custom_output_model.eval()
    # custom_output_model(img)
    # x = torch.zeros((1, 3, 640, 640))
    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    # output = ['bbox_out', 'class_out']

    torch.onnx._export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names= [args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    # torch.onnx._export(
    #     custom_output_model,
    #     dummy_input,
    #     args.output_name,
    #     input_names=[args.input],
    #     output_names= ["bbox_out", "class_out"],
    #     dynamic_axes= None,
    #     opset_version=args.opset,
    # )
    # logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        # use onnx-simplifier to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
