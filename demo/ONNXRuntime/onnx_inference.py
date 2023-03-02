#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime
import torch

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES, ROBOTIS_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/home/robotis-workstation3/ai/YOLOX-ROBOTIS/yolox_s_pretrained_robotis_custom_test_normal_model.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='/home/robotis-workstation3/ai/YOLOX-ROBOTIS/test/2023_2_7_11_21_11.jpg',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    boxes = output[0]
    scores = output[1]


    print(output)
    # predictions = demo_postprocess(output, input_shape, p6=args.with_p6)[0]

    # boxes = output[:, :4]
    # scores = output[:, 4:5] * output[:, 5:]

    # boxes_xyxy = np.ones_like(boxes)
    # boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    # boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    # boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    # boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    # scores = scores.reshape(1, -1)[0]
    # boxes_xyxy = boxes / ratio
    # boxes_xyxy /= ratio
    boxes = boxes / ratio
    # boxes_xyxy = (boxes /  ratio).reshape(1, -1, 4)[0]
    dets = multiclass_nms(boxes, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=ROBOTIS_CLASSES)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
