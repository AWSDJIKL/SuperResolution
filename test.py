# -*- coding: utf-8 -*-
'''
单元测试
'''
# @Time    : 2021/8/5 15:43
# @Author  : LINYANZHEN
# @File    : model_test.py


import numpy as np
from PIL import Image
import h5py

if __name__ == '__main__':
    path = "checkpoint/test.jpg"
    upscale_factor = 3
    hr = Image.open(path).convert('RGB')
    hr_width = (hr.width // upscale_factor) * upscale_factor
    hr_height = (hr.height // upscale_factor) * upscale_factor
    hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
    lr = hr.resize((hr_width // upscale_factor, hr_height // upscale_factor), resample=Image.BICUBIC)
    hr = np.array(hr).astype(np.float32)
    lr = np.array(lr).astype(np.float32)
    print(hr.shape)  # (h,w,c)
    patch_size = 17
    stride = 13
    lr_patches = []
    hr_patches = []
    # h
    for i in range(0, lr.shape[0] - patch_size + 1, stride):
        # w
        for j in range(0, lr.shape[1] - patch_size + 1, stride):
            lr_patches.append(lr[i:i + patch_size, j:j + patch_size])
            hr_patches.append(hr[i * upscale_factor:i * upscale_factor + patch_size * upscale_factor,
                              j * upscale_factor:j * upscale_factor + patch_size * upscale_factor])
    print(lr_patches[0].shape)
    print(hr_patches[0].shape)
