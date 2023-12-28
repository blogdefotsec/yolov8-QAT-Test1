# Knowledge Distillation on YOLOv8 Detection Model
Implementation of **Focal and Global Knowledge Distillation for Detectors**(https://arxiv.org/abs/2111.11837) on YOLOv8
## Usage
- Multi GPU
```
python -m torch.distributed.run --nproc_per_node {} distill_train.py --dataset-dir {} --dataset-config {} --teacher-config {} --teacher-weight {} --student-config {} --student-weight
```
## Results
Experiment Ongoing