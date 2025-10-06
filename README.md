# Vision Transformer Research Framework (PyTorch + TIMM)

A modular research framework built in **PyTorch** for exploring **Vision Transformer (ViT) backbones** across **image classification**, **object detection**, and **semantic segmentation** tasks.  
Supports **timm** backbones, custom task-specific heads, and YAML-based configuration for flexible experimentation.

---

## Overview

This repository is designed for:
- Research and prototyping with Vision Transformer architectures.
- Comparing different backbones from the [`timm`](https://github.com/huggingface/pytorch-image-models) library.
- Custom head design for classification, detection, and segmentation.
- Reproducible experiments via YAML configuration.
- Logging and checkpointing with TensorBoard.

---

## Repository Structure

```
vision-transformer-research/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ cls_resnet50.yaml
│  ├─ det_vit_fpn.yaml
│  └─ seg_vit.yaml
├─ datasets/
│  ├─ __init__.py
│  ├─ classification_dataset.py
│  ├─ detection_dataset.py   # COCO / Pascal-style wrapper
│  └─ segmentation_dataset.py   
├─ models/
│  ├─ __init__.py
│  ├─ backbones.py           # wrappers around timm models -> features_only
│  ├─ classification.py      # classification head + model builder
│  ├─ detection.py           # wrapper to create torchvision detection model with timm backbone
│  └─ segmentation.py        # wrapper to create torchvision segmentation model with timm 
├─ utils/
│  ├─ transforms.py
│  ├─ train_utils.py
│  └─ metrics.py
├─ train/
│  ├─ train_classification.py
│  ├─ train_detection.py
│  └─ train_segmentation.py
└─ experiments/
   └─ logs, checkpoints, tensorboard

## Installation

```bash
# Clone the repo
git clone https://github.com/gokulkrishna-sys/vision-transformer-research.git
cd vision-transformer-research

# Create and activate virtual environment
conda create -n vit_research python=3.10 -y
conda activate vit_research

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
- torch, torchvision
- timm
- pyyaml
- opencv-python
- torchmetrics
- tensorboard

---

## How to Run

### Classification
```bash
python train/train_classification.py --config configs/cls_vit_base.yaml
```

### Object Detection
```bash
python train/train_detection.py --config configs/det_vit_fpn.yaml
```

### Semantic Segmentation
```bash
python train/train_segmentation.py --config configs/seg_vit.yaml
```

All experiment results (checkpoints, logs, TensorBoard runs) are saved automatically under:
```
experiments/<experiment_name>/
```

---

## Modifying the Framework

### Change Backbone
All backbones are loaded from [timm](https://github.com/huggingface/pytorch-image-models):
```yaml
backbone: vit_base_patch16_224
```
You can replace this with any other model name supported by timm, e.g.
`convnext_base`, `swin_large_patch4_window7_224`, etc.

---

### Change Dataset
Each config defines dataset paths:
```yaml
train_dir: ./data/imagenet/train
val_dir: ./data/imagenet/val
```
Replace them with your custom dataset paths.  
For detection and segmentation, COCO and Pascal VOC formats are supported by default.

---

### Change Head Architecture
Modify the respective file under `models/`:
- `classification_head.py` → MLP, global pooling, dropout tuning
- `detection_head.py` → FasterRCNN, RetinaNet, custom heads
- `segmentation_head.py` → FCN, DeepLab, U-Net style decoders

Each head receives features from a timm backbone via `model_builder.py`.

---

### Add a New Task
You can easily extend to new tasks (e.g., self-supervised, depth estimation):
1. Create `models/new_task_head.py`
2. Add dataset loader in `utils/datasets.py`
3. Add `train/train_new_task.py`
4. Write a YAML config under `configs/`

---

## Logging & Checkpoints

- TensorBoard logs: `experiments/<exp_name>/runs/`
- Model checkpoints: automatically saved in the same directory
- Resume training: supported by setting
  ```yaml
  resume_from: ./experiments/<exp_name>/checkpoint.pth
  ```

---

## Example: Custom Classification Experiment

1. Edit your YAML:
```yaml
backbone: swin_base_patch4_window7_224
num_classes: 10
train_dir: ./data/cifar10/train
val_dir: ./data/cifar10/val
epochs: 100
```

2. Run:
```bash
python train/train_classification.py --config configs/cls_vit_base.yaml
```

3. Visualize:
```bash
tensorboard --logdir experiments/
```

---

## Research Ideas to Explore

- Effect of backbone choice (ViT, Swin, ConvNeXt) on downstream tasks  
- Transfer learning with frozen vs fine-tuned encoders  
- Hybrid CNN-ViT backbones for detection  
- Lightweight ViT models for segmentation  
