from ultralytics import YOLO
from ultralytics.qat.nvidia_tensorrt.qat_nvidia_trainer import QuantizationTrainer
from ultralytics import settings
import argparse

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, callbacks, checks, emojis, yaml_load

def _reset_ckpt_args(args):
    """Reset arguments when loading a PyTorch model."""
    include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
    return {k: v for k, v in args.items() if k in include}

def train(args):
    # Load a mode
    model, ckpt = attempt_load_one_weight('yolov8n.pt')
    task = model.args['task']
    overrides = model.args = _reset_ckpt_args(model.args)
    overrides['model'] = args.model_config
    overrides['data'] = args.data_config
    trainer = QuantizationTrainer(cfg = DEFAULT_CFG_DICT.copy(), overrides=overrides)
    trainer.model = trainer.get_model(args.model_config,
                                       args.pretrained_weight)

    results = trainer.train()
    return results
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default = 'yolov8s.yaml')
    parser.add_argument("--pretrained-weight", type=str, default = 'yolo8s.pt')
    parser.add_argument("--data-config", type=str, default = 'coco.yaml')
    args = parser.parse_args()
    train(args)

