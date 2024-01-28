# YOLOv8 QAT

Original Ultralytics compatible. (You can pretrain your model befor QAT as original way using this repository)
## Usage

Install editable package in your environment by `pip install -e .`

### Using pytorch native quantization API

```bash
python qat_pytorch.py \
--model-config ${model_config_yaml_file} \
--pretrained-weight ${path_to_your_pretrained_weight} \
--data-config ${path_to_your_data_config_file}
```
### Using `pytorch_quantization` package from nvidia
You need to install `pytorch_quantization` package

```bash
python qat_nvidia.py \
--model-config ${model_config_yaml_file} \
--pretrained-weight ${path_to_your_pretrained_weight} \
--data-config ${path_to_your_data_config_file}
```

## TODO
- end-to-end export to TensorRT engine(when using pytorch_quantization)
- code refactoring 
- find other ways to improve mAP after QAT

## References
https://medium.com/@DeeperAndCheaper/quantization-yolov8-qat-x2-speed-up-on-your-jetson-orin-nano-2-how-to-achieve-the-best-qat-8077ac0a167b

https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training