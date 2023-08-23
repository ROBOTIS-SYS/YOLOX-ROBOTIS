#!/bin/bash

if [ "$1" == "train" ]; then
  echo "Model Train Start!"
  python3 tools/train.py -f exps/example/yolox_voc/yolox_voc_l_thyssen.py -c ./model/yolox_l.pth -d 1 -b 8 -o --cache
fi

if [ "$1" == "onnx" ]; then
  echo "ONNX export Start!"
  python3 tools/export_onnx.py -f exps/example/yolox_voc/yolox_voc_l_thyssen.py -c /home/robotis-workstation3/ai/YOLOX-ROBOTIS/YOLOX_outputs/yolox_voc_l_thyssen/best_ckpt.pth --output-name=yolox-thyssen_indicator_v2.onnx
fi

if [ "$1" == "demo" ]; then
  echo "Demo Start!"
  python3 tools/demo.py image -f exps/example/yolox_voc/yolox_voc_l_thyssen.py -c ./YOLOX_outputs/yolox_voc_l_thyssen/best_ckpt.pth --path assets/dog.jpg --conf 0.6 --nms 0.5 --tsize 640 --device gpu --save_result
  # ex) path : /home/robotis-workstation3/ai/dataset/robotis_thyssen/1/sample/*.jpg
fi

