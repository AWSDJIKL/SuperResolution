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

python train.py --save_dir "checkpoint" --upscale_factor 3 --lr 1e-3 --epoch 100 --batch-size 16 --num-workers 8
