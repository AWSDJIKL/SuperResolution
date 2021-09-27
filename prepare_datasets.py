# -*- coding: utf-8 -*-
'''
准备数据集
'''
# @Time    : 2021/8/11 10:34
# @Author  : LINYANZHEN
# @File    : prepare_datasets.py
import gzip
import os
import shutil
import tarfile
import time
import zipfile
import psutil
import sys
import h5py
import numpy as np
import wget
from PIL import Image
import datasets

link_list = {
    "set5": "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip",
    "set14": "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip",
    "Urban100": "https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip",
    "BSD500": "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
    "DIV2K_train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def uncompress(src_file, output_dir=None):
    '''
    解压文件，默认解压到文件所在目录下

    :param src_file: 要解压的文件
    :param output_dir: 解压输出路径
    :return:
    '''

    # 获取文件后缀名，判断压缩格式
    file_name, file_format = os.path.splitext(src_file)
    # 创建解压路径
    if output_dir:
        os.mkdir(output_dir)
    else:
        file_path, _ = os.path.split(src_file)
        # output_dir = os.path.join(file_path, file_name)
        output_dir = file_path
        # os.mkdir(output_dir)
        print(output_dir)
    if file_format in ('.tgz', '.tar'):
        tar = tarfile.open(src_file)
        names = tar.getnames()
        for name in names:
            tar.extract(name, output_dir)
        tar.close()
    elif file_format == '.zip':
        zip_file = zipfile.ZipFile(src_file)
        for names in zip_file.namelist():
            zip_file.extract(names, output_dir)
        zip_file.close()
    elif file_format == '.gz':
        f_name = output_dir + '/' + os.path.basename(src_file)
        g_file = gzip.GzipFile(src_file)
        open(f_name, "w+").write(g_file.read())
        g_file.close()
    else:
        print('文件格式不支持或者不是压缩文件')
        return


def download_datasets(dataset_path):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    for name, link in link_list.items():
        print(name)
        print(link)
        output_path = os.path.join(dataset_path, name)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        file_name = wget.download(link, output_path)
        print(file_name)
        uncompress(file_name)


def prepare_train_h5py(image_path_list, upscale_factor, output_dir, crop_image=False, patch_size=17, stride=13,
                       block_size=1 * 1024 * 1024):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    block_count = 0
    h5file_path = os.path.join(output_dir, "{}.h5".format(block_count))
    h5_file = h5py.File(h5file_path, 'w')
    lr_list = []
    hr_list = []
    # 检测内存占用
    memory = psutil.virtual_memory()

    for path in image_path_list:
        hr = Image.open(path).convert('RGB')
        hr_width = (hr.width // upscale_factor) * upscale_factor
        hr_height = (hr.height // upscale_factor) * upscale_factor
        hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        lr = hr.resize((hr_width // upscale_factor, hr_height // upscale_factor), resample=Image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = np.swapaxes(np.swapaxes(hr, 0, 2), 1, 2)
        lr = np.swapaxes(np.swapaxes(lr, 0, 2), 1, 2)
        # hr = np.swapaxes(hr, 0, 2)
        # lr = np.swapaxes(lr, 0, 2)
        if crop_image:
            # h
            for i in range(0, lr.shape[1] - patch_size + 1, stride):
                # w
                for j in range(0, lr.shape[2] - patch_size + 1, stride):
                    lr_list.append(lr[:,i:i + patch_size, j:j + patch_size])
                    hr_list.append(hr[:,i * upscale_factor:i * upscale_factor + patch_size * upscale_factor,
                                   j * upscale_factor:j * upscale_factor + patch_size * upscale_factor])
        else:
            lr_list.append(lr)
            hr_list.append(hr)

        # print(sys.getsizeof(lr_list) + sys.getsizeof(hr_list))
        # 检查内存使用情况，若超过指定block_size则分块
        if sys.getsizeof(lr_list) + sys.getsizeof(hr_list) > block_size:
            # 保存
            lr_list = np.array(lr_list)
            hr_list = np.array(hr_list)
            h5_file.create_dataset('lr', data=lr_list)
            h5_file.create_dataset('hr', data=hr_list)
            h5_file.close()
            block_count += 1
            # 开启下一个分块
            h5file_path = os.path.join(output_dir, "{}.h5".format(block_count))
            h5_file = h5py.File(h5file_path, 'w')
            lr_list = []
            hr_list = []

    # 保存最后的分块
    lr_list = np.array(lr_list)
    hr_list = np.array(hr_list)
    h5_file.create_dataset('lr', data=lr_list)
    h5_file.create_dataset('hr', data=hr_list)
    h5_file.close()
    block_size += 1


def prepare_val_h5py(image_path_list, upscale_factor, output_dir, block_size=5 * 1024 * 1024):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    block_count = 0
    h5file_path = os.path.join(output_dir, "{}.h5".format(block_count))
    h5_file = h5py.File(h5file_path, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    # 检测内存占用
    memory = psutil.virtual_memory()
    for index, path in enumerate(image_path_list):
        hr = Image.open(path).convert('RGB')
        hr_width = (hr.width // upscale_factor) * upscale_factor
        hr_height = (hr.height // upscale_factor) * upscale_factor
        hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        lr = hr.resize((hr_width // upscale_factor, hr_height // upscale_factor), resample=Image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = np.swapaxes(np.swapaxes(hr, 0, 2), 1, 2)
        lr = np.swapaxes(np.swapaxes(lr, 0, 2), 1, 2)
        # hr = np.swapaxes(hr, 0, 2)
        # lr = np.swapaxes(lr, 0, 2)
        lr_group.create_dataset(str(index), data=lr)
        hr_group.create_dataset(str(index), data=hr)

        # 检查内存使用情况，若超过指定block_size则分块
        if sys.getsizeof(lr_group) + sys.getsizeof(hr_group) > block_size:
            # 保存
            h5_file.close()
            block_size += 1
            # 开启下一个分块
            h5file_path = os.path.join(output_dir, "{}.h5".format(block_count))
            h5_file = h5py.File(h5file_path, 'w')
            lr_group = h5_file.create_group('lr')
            hr_group = h5_file.create_group('hr')
    # 保存最后的分块
    h5_file.close()
    block_size += 1


if __name__ == '__main__':
    # print("开始下载数据集")
    # dataset_path = "dataset"
    # download_datasets(dataset_path)
    # print("所有数据集下载完成")

    print("开始集合数据集")
    train_image_list = datasets.get_train_image_list()
    val_image_list = datasets.get_val_image_list()
    print("训练集共{}张图片".format(len(train_image_list)))
    print("测试集共{}张图片".format(len(val_image_list)))
    train_set_output_dir = "dataset/x3_train"
    val_set_output_dir = "dataset/x3_val"
    prepare_train_h5py(train_image_list, 3, train_set_output_dir, crop_image=True)
    prepare_val_h5py(val_image_list, 3, val_set_output_dir)
    print("数据集集合完毕")
