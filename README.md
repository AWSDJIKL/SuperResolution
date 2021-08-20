# SuperResolution

## Requirements

- PyTorch
- Numpy
- Pillow
- h5py
- wget
- tarfile
- zipfile

## Prepare

python prepare_datasets.py

## Train

python train.py --upscale_factor 3 --lr 1e-3 --epoch 100

## 目前主要问题

棋盘效应过于严重

### 猜测原因

- 卷积过程中使用了padding
- 与损失函数使用了卷积有关

## 实验记录