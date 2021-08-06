# -*- coding: utf-8 -*-
'''
工具函数
'''
# @Time    : 2021/8/5 15:08
# @Author  : LINYANZHEN
# @File    : utils.py
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
from torchvision.transforms import ToTensor


def calaculate_psnr(img1, img2):
    '''
    计算两张图片之间的PSNR误差

    :param img1:
    :param img2:
    :return:
    '''
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2)).item()


def load_image_RGB(image_path):
    '''
    正常加载图片

    :param image_path: 图片路径
    :return:
    '''
    image = Image.open(image_path).convert("RGB")
    return image


def load_image_ycbcr(image_path):
    '''
    以YCbCr格式加载图片并按通道分割

    :param image_path: 图片路径
    :return:
    '''
    y, cb, cr = Image.open(image_path).convert('YCbCr').split()
    y = Variable(ToTensor()(y))
    cb = Variable(ToTensor()(cb))
    cr = Variable(ToTensor()(cr))
    return y, cb, cr


def lr_transform(img_size, upscale_factor):
    '''

    :param img_size: 原图大小
    :param upscale_factor:
    :return:
    '''
    new_size = [i // upscale_factor for i in img_size]
    # print("original size =", img_size)
    # print("new size =", new_size)
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.Resize(new_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])


def hr_transform(img_size):
    '''

    :param img_size:
    :return:
    '''
    return transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def tensor_to_image(tensor):
    return transforms.ToPILImage()(tensor)


class AverageMeter(object):
    '''
    记录数据并计算平均数
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=1):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count


def prepare_super_resolution_loaders(dataset_list):
    train_loader_list = []
    val_loader_list = []
    for dataset in dataset_list:
        train_loader, val_loader = get_super_resolution_dataloader(dataset)
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
    return train_loader_list, val_loader_list


def get_super_resolution_dataloader(dataset_name):
    if dataset_name == "Aircraft":
        from dataset import Aircraft
        data_dir = "D:/Dataset/fgvc-aircraft-2013b/data/images"
        train_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_train.txt"
        val_labels = "D:/Dataset/fgvc-aircraft-2013b/data/images_val.txt"

        train_loader = torch.utils.data.DataLoader(Aircraft.AircraftDataset(data_dir, train_labels, upscale_factor=3))
        val_loader = torch.utils.data.DataLoader(Aircraft.AircraftDataset(data_dir, val_labels, upscale_factor=3))

        return train_loader, val_loader
