# -*- coding: utf-8 -*-
'''
超分辨通用数据集类
'''
# @Time    : 2021/8/5 15:27
# @Author  : LINYANZHEN
# @File    : datasets.py
import os

from torch.utils.data import Dataset
import h5py
from utils import load_image_RGB, lr_transform, hr_transform
from torch.utils.data import DataLoader


class GeneralH5pyDataset(Dataset):
    def __init__(self, data_file_path):
        super(GeneralH5pyDataset, self).__init__()
        self.data_file_path = data_file_path

    def __getitem__(self, index):
        with h5py.File(self.data_file_path, 'r') as f:
            return f['lr'][index], f['hr'][index]

    def __len__(self):
        with h5py.File(self.data_file_path, 'r') as f:
            return len(f['lr'])


class GeneralRGBDataset(Dataset):
    def __init__(self, img_list, upscale_factor=3, loader=load_image_RGB):
        super(GeneralRGBDataset, self).__init__()
        self.img_list = img_list
        self.upscale_factor = upscale_factor
        self.loader = loader

    def __getitem__(self, index):
        img = self.loader(self.img_list[index])
        width = (img.width // self.upscale_factor) * self.upscale_factor
        height = (img.height // self.upscale_factor) * self.upscale_factor
        lr = lr_transform((height, width), self.upscale_factor)(img)
        hr = hr_transform((height, width))(img)
        return lr, hr

    def __len__(self):
        return len(self.img_list)


def get_train_image_list():
    file_dirs = [
        "dataset/BSD500/BSR/BSDS500/data/images",
        # "dataset/DIV2K_train/DIV2K_train_HR",
        # "dataset/DIV2K_valid/DIV2K_valid_HR",
    ]
    image_path_list = []
    for i in file_dirs:
        for root, dirs, files in os.walk(i):
            print(1)
            for image_path in files:
                if not ".db" in image_path:
                    image_path_list.append(os.path.join(root, image_path))
    return image_path_list


def get_val_image_list():
    file_dirs = [
        "dataset/set5/Set5/image_SRF_2",
        "dataset/set14/Set14/image_SRF_2",
        "dataset/Urban100/image_SRF_2",
    ]
    image_path_list = []
    for i in file_dirs:
        for root, dirs, files in os.walk(i):
            for image_path in files:
                if "_HR" in image_path:
                    image_path_list.append(os.path.join(root, image_path))
    return image_path_list


def get_super_resolution_dataloader(args):
    train_list = get_train_image_list()
    val_list = get_val_image_list()
    train_loader = DataLoader(GeneralRGBDataset(train_list, upscale_factor=args.upscale_factor),
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    val_loader = DataLoader(GeneralRGBDataset(val_list, upscale_factor=args.upscale_factor))
    return train_loader, val_loader
