# -*- coding: utf-8 -*-
'''
将图片数据整理好打包成h5py文件方便读取
'''
# @Time    : 2021/8/5 15:36
# @Author  : LINYANZHEN
# @File    : prepare.py
import h5py
import numpy as np
from PIL import Image


def prepare_h5py(image_path_list, upscale_factor, output_path, crop_image=False, patch_size=72, stride=30):
    output_path = output_path.replace(".h5", "_x{}.h5".format(upscale_factor))
    h5_file = h5py.File(output_path, 'w')

    lr_list = []
    hr_list = []
    for path in image_path_list:
        hr = Image.open(path).convert('RGB')
        hr_width = (hr.width // upscale_factor) * upscale_factor
        hr_height = (hr.height // upscale_factor) * upscale_factor
        hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        lr = hr.resize((hr_width // upscale_factor, hr_height // upscale_factor), resample=Image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        if crop_image:
            # h
            for i in range(0, lr.shape[0] - patch_size + 1, stride):
                # w
                for j in range(0, lr.shape[1] - patch_size + 1, stride):
                    lr_list.append(lr[i:i + patch_size, j:j + patch_size])
                    hr_list.append(hr[i * upscale_factor:i * upscale_factor + patch_size * upscale_factor,
                                   j * upscale_factor:j * upscale_factor + patch_size * upscale_factor])
        else:
            lr_list.append(lr)
            hr_list.append(hr)

    lr_list = np.array(lr_list)
    hr_list = np.array(hr_list)

    h5_file.create_dataset('lr', data=lr_list)
    h5_file.create_dataset('hr', data=hr_list)

    h5_file.close()
