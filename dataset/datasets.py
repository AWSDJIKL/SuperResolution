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


class GeneralDataset(Dataset):
    def __init__(self, data_file_path):
        super(GeneralDataset, self).__init__()
        self.data_file_path = data_file_path

    def __getitem__(self, index):
        with h5py.File(self.data_file_path, 'r') as f:
            return f['lr'][index], f['hr'][index]

    def __len__(self):
        with h5py.File(self.data_file_path, 'r') as f:
            return len(f['lr'])
