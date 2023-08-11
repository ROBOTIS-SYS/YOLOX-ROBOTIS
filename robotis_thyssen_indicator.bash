#!/bin/bash
# python3 tools/demo.py image -f exps/default/yolox_s_robotis_thyssen_indicator.py -c ./YOLOX_outputs/yolox_s_robotis_thyssen/best_ckpt.pth --path assets/dog.jpg --conf 0.6 --nms 0.5 --tsize 640 --device gpu --save_result
# python3 tools/train.py -f exps/default/yolox_s_robotis_thyssen_indicator.py -c ./model/yolox_s.pth -d 1 -b 16 -o --cache
# python3 tools/train.py -f exps/default/yolox_s_robotis_thyssen_indicator.py -c ./YOLOX_outputs/yolox_s_robotis_thyssen/last_epoch_ckpt.pth -d 1 -b 16 -o --cache --resume
# python3 tools/export_onnx.py -f exps/default//yolox_s_robotis_thyssen_indicator.py -c ./YOLOX_outputs/yolox_s_robotis_thyssen_indicator.py/best_ckpt.pth --output-name=yolox-thyssen_v2.onnx

if [ "$1" == "train" ]; then
  echo "Model Train Start!"
  python3 tools/train.py -f exps/default/yolox_s_robotis_thyssen_indicator.py -c ./model/yolox_s.pth -d 1 -b 16 -o --cache
fi

if [ "$1" == "onnx" ]; then
  echo "ONNX export Start!"
  python3 tools/export_onnx.py -f exps/default//yolox_s_robotis_thyssen_indicator.py -c /home/robotis-workstation3/ai/YOLOX-ROBOTIS/YOLOX_outputs/yolox_s_robotis_thyssen_indicator/best_ckpt.pth --output-name=yolox-thyssen_indicator_v2.onnx
fi

if [ "$1" == "demo" ]; then
  echo "Demo Start!"
  python3 tools/demo.py image -f exps/default/yolox_s_robotis_thyssen_indicator.py -c ./YOLOX_outputs/yolox_s_robotis_thyssen_indicator/best_ckpt.pth --path assets/dog.jpg --conf 0.6 --nms 0.5 --tsize 640 --device gpu --save_result
fi

