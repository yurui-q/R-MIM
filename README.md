## R-MIM
This repository is the official [PyTorch](http://pytorch.org) implementation of R-MIM.

## 2 Requirements
- Python 3.6
- [PyTorch](http://pytorch.org) install = 1.6.0
- torchvision install = 0.7.0
- CUDA 10.1

## 3 Pretraining
We release a demo for the R-MIM self-supervised learning approach. The model is based on ViT-small architecture, pretrained for 300 epochs.

To train R-MIM on a single node with 4 gpus for 300 epochs, run:
```
DATASET_PATH="path/to/ImageNet1K/train"
EXPERIMENT_PATH="path/to/experiment"

python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
    --output_dir ${EXPERIMENT_PATH} \
    --log_dir ${EXPERIMENT_PATH} \
    --batch_size 128 --accum_iter 8 \
    --model mae_vit_small_patch16_dec512d8b \
    --input_size 224 \
    --epochs 300 \
    --norm_pix_loss \
    --warmup_epochs 15 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path ${DATASET_PATH} \
    --kernel RBF \
    --gamma 3e-7 
```

## 4 Finetuning
To finetune the ViT-small pretrained via R-MIM for 100 epochs, run:
```
DATASET_PATH="path/to/ImageNet1K/train"
EXPERIMENT_PATH="path/to/experiment"
CHECKPOINT_PATH="path/to/checkpoint.pth"

python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
    --output_dir ${EXPERIMENT_PATH} \
    --log_dir ${EXPERIMENT_PATH} \
    --batch_size 128 --accum_iter 2 \
    --model vit_small_patch16 \
    --nb_classes 1000 \
    --epochs 100 \
    --warmup_epochs 5 \
    --blr 1e-3 \
    --weight_decay 0.05 \
    --layer_decay 0.75 \
    --drop_path 0.1 --reprob 0.25 \
    --mixup 0.8 --cutmix 1.0 \
    --aa rand-m9-mstd0.5-inc1 \
    --smoothing 0.1 \
    --dist_eval \
    --data_path ${DATASET_PATH} \
    --betas 0.9 0.999\
    --finetune ${CHECKPOINT_PATH}
```

## 5 Linear Probe
To train a linear classifier on frozen features out of deep network pretrained via R-MIM for 90 epochs, run:
```
DATASET_PATH="path/to/ImageNet1K/train"
EXPERIMENT_PATH="path/to/experiment"
CHECKPOINT_PATH="path/to/checkpoint.pth"

python -m torch.distributed.launch --nproc_per_node=4 main_linprobe.py \
    --output_dir ${EXPERIMENT_PATH} \
    --log_dir ${EXPERIMENT_PATH} \
    --batch_size 512 --accum_iter 4 \
    --model vit_small_patch16 \
    --nb_classes 1000 \
    --epochs 90 \
    --warmup_epochs 10 \
    --min_lr 0. \
    --blr 0.1 \
    --weight_decay 0. \
    --dist_eval \
    --data_path ${DATASET_PATH} \
    --finetune ${CHECKPOINT_PATH}
```