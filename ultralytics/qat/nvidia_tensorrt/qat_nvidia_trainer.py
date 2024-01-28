# Ultralytics YOLO 🚀, AGPL-3.0 license
# Implementation of https://medium.com/@DeeperAndCheaper/quantization-achieve-accuracy-drop-to-near-zero-yolov8-qat-x2-speed-up-on-your-jetson-orin-nano-e178c4d8a5e3

from copy import copy
import math
import time
import numpy as np
import torch 
import torch.nn as nn
import warnings
from torch import distributed as dist
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM, __version__, colorstr, callbacks
from ultralytics.utils.torch_utils import  de_parallel, is_parallel, ModelEMA
from copy import deepcopy
from datetime import datetime 
import logging 
from torch import nn, optim
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_imgsz
from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA
from pytorch_quantization import quant_modules
from .quant_ops import quant_module_change
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from tqdm import tqdm
import torch.distributed as dist


def cal_model(model, data_loader, device, num_batch=1024):
    num_batch = num_batch
    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)
        
    def collect_stats(model, data_loader, device, num_batch=1024):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                # imgs = datas[0].to(device, non_blocking=True).float() / 255.0
                imgs = datas['img'].to(device, non_blocking=True).float() / 255.0
                model(imgs)

                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, data_loader, device, num_batch=num_batch)
    compute_amax(model,  method="percentile", percentile=99.99)


class QuantizationTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.qat = 'nvidia'

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """_summary_

        Args:
            cfg (_type_, optional): Student model configuration. Defaults to None.
            weights (_type_, optional): Student weight configuration. Defaults to None.
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=16 * 2, rank=-1, mode='val')
        quant_modules.initialize()
        if RANK in (-1, 0):
            # Do calibration x
            _temp_model= DetectionModel(cfg = cfg, ch = 3, nc=  self.data['nc'], verbose = False)
            quant_module_change(_temp_model)
            if weights is not None:
                _temp_model.load_state_dict(torch.load(weights)['model'].state_dict())
            _temp_model.cuda()
            cal_model(_temp_model, self.test_loader, 'cuda')
            torch.save(_temp_model.state_dict(), self.wdir / f'_temp_calibration.pt')
        if RANK != -1:
            dist.barrier()
        model = DetectionModel(cfg = cfg, ch = 3, nc=  self.data['nc'], verbose = False)
        quant_module_change(model )
        model.load_state_dict(torch.load(self.wdir / f'_temp_calibration.pt'))
        
        return model
    
    def setup_model(self):
        """Do nothing here - directly call get_model from training code"""
        return None

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max= self.epochs * 1.0, eta_min=self.args.lr0 * 0.01)

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        self.args.warmup_epochs = 0
        self.epochs = self.epochs // 5
        self.args.lr0 /= 100

        # Model
        self.run_callbacks('on_pretrain_routine_start')

        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = self.args.freeze if isinstance(
            self.args.freeze, list) else range(self.args.freeze) if isinstance(self.args.freeze, int) else []
        always_freeze_names = ['.dfl']  # always freeze these layers
        freeze_layer_names = [f'model.{x}.' for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad:
                LOGGER.info(f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                            'See ultralytics.engine.trainer for customization of frozen layers.')
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(False).to(self.device)  # Must be FALSE in QAT setting
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations)
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks('on_pretrain_routine_end')

