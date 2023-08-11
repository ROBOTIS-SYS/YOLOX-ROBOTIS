#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .robotis_classes import ROBOTIS_CLASSES
from .robotis_local_classes import LOCATION_DETECT_CLASSES
from .robotis_dynamic_classes import DYNAMIC_CLASSES
from .robotis_dynamic_head_body_classes import DYNAMIC_BODY_HEAD_CLASSES
from .robotis_thyssen_indicator import ROBOTIS_THYSSEN_INDICATOR
from .robotis_thyssen import ROBOTIS_THYSSEN
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection
