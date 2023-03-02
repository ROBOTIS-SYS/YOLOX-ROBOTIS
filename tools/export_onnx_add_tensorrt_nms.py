#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import onnx_graphsurgeon as gs
import onnx

import numpy as np


def create_attrs(input_h, input_w, topK, keepTopK):
    attrs = {}
    attrs["shareLocation"] = 1
    attrs["backgroundLabelId"] = -1
    attrs["numClasses"] = 1
    attrs["topK"] = topK
    attrs["keepTopK"] = keepTopK
    attrs["scoreThreshold"] = 0.25
    attrs["iouThreshold"] = 0.6
    attrs["isNormalized"] = False
    attrs["clipBoxes"] = False

    attrs["plugin_version"] = "1"

    return attrs

@logger.catch
def main():
    graph = gs.import_onnx(onnx.load("./yolox_s_pretrained_robotis_custom.onnx"))
    batch_size = graph.inputs[0].shape[0]
    input_h = graph.inputs[0].shape[2]
    input_w = graph.inputs[0].shape[3]
    tensors = graph.tensors()

    boxes_tensor = tensors["bbox_out"] # model output name
    confs_tensor = tensors["class_out"] # model output name
    topK = 100
    keepTopK = 50

    num_detections =gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1])
    numsed_boxes = gs.Variable(name="nmsed_boxes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK, 4])
    nmsed_scores = gs.Variable(name="nmsed_scroes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK, 1])
    nmsed_classes = gs.Variable(name="nmsed_classes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK])

    new_outputs = [num_detections, numsed_boxes, nmsed_scores, nmsed_classes]

    nms_node = gs.Node(
        op="BatchedNMS_TRT",
        attrs= create_attrs(input_h, input_w, topK, keepTopK),
        inputs=[boxes_tensor, confs_tensor],
        outputs=new_outputs)

    graph.nodes.append(nms_node)
    graph.outputs = new_outputs

    graph = graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph), 'yolox_s_robotis_batched_nms.onnx')

if __name__ == "__main__":
    main()
