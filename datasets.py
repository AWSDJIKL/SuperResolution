# -*- coding: utf-8 -*-
'''
超分辨通用数据集类
'''
# @Time    : 2021/8/5 15:27
# @Author  : LINYANZHEN
# @File    : datasets.py

from torch.utils.data import Dataset
import h5py
import numpy as np
import utils


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
    def __init__(self, img_list, upscale_factor=3, loader=utils.load_image_RGB):
        super(GeneralRGBDataset, self).__init__()
        self.img_list = img_list
        self.upscale_factor = upscale_factor
        self.loader = loader

    def __getitem__(self, index):
        img = self.loader(self.img_list[index])
        width = (img.width // self.upscale_factor) * self.upscale_factor
        height = (img.height // self.upscale_factor) * self.upscale_factor
        lr = utils.lr_transform((height, width), self.upscale_factor)(img)
        hr = utils.hr_transform((height, width))(img)
        return lr, hr

    def __len__(self):
        return len(self.img_list)
