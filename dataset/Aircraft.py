# -*- coding: utf-8 -*-
'''
Aircraft数据集导入(用于超分辨率)
飞机数据集本应作为分类数据集使用，这里仅用于测试
'''
# @Time    : 2021/8/5 15:08
# @Author  : LINYANZHEN
# @File    : Aircraft.py
import os
import os.path
import torch
import torch.utils.data
from torchvision import transforms
import utils
from PIL import Image
from dataset import prepare


class AircraftDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_list_file, upscale_factor=3, loader=utils.load_image_RGB):
        '''
        Aircraft数据集

        :param data_dir:
        :param img_list_file:
        :param upscale_factor:
        :param loader:
        '''
        super(AircraftDataset, self).__init__()
        self.img_list = []
        with open(img_list_file, "r") as file:
            for img in file.readlines():
                self.img_list.append(os.path.join(data_dir, img.strip() + ".jpg"))
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


def prepare_dataloader():
    train_lr_img_dir = "D:/Dataset/fgvc-aircraft-2013b/data/train/lr"
    train_hr_img_dir = "D:/Dataset/fgvc-aircraft-2013b/data/train/hr"
    test_lr_img_dir = "D:/Dataset/fgvc-aircraft-2013b/data/test/lr"
    test_hr_img_dir = "D:/Dataset/fgvc-aircraft-2013b/data/test/hr"

    train_loader = torch.utils.data.DataLoader(AircraftDataset(train_lr_img_dir, train_hr_img_dir))
    val_loader = torch.utils.data.DataLoader(AircraftDataset(test_lr_img_dir, test_hr_img_dir))
    print(train_loader.dataset[0])
    print(val_loader.dataset[0])
    return


def prepare_data_file(root_dir, img_list_file, output_path, upscale_factor):
    img_list = []
    with open(img_list_file, "r") as file:
        for img in file.readlines():
            img_list.append(os.path.join(root_dir, img.strip() + ".jpg"))
    prepare.prepare_h5py(img_list, upscale_factor, output_path)


if __name__ == '__main__':
    data_dir = "D:/Dataset/fgvc-aircraft-2013b/data/images"
    train_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_train.txt"
    val_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_val.txt"
    train_data_path = "Aircraft_train.h5"
    val_data_path = "Aircraft_val.h5"
    prepare_data_file(data_dir, train_labels, train_data_path, 3)
    prepare_data_file(data_dir, val_labels, val_data_path, 3)
