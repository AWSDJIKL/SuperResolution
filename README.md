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

python PerceptualLoss/train.py --upscale_factor 3 --lr 1e-3 --epoch 100 --batch_size 16 --num_workers 8
