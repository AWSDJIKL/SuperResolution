# -*- coding: utf-8 -*-
'''
测试模型超分辨效果
'''
# @Time    : 2021/8/2 23:01
# @Author  : LINYANZHEN
# @File    : test.py
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
import utils
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    upscale_factor = 3
    test_image_path = "test/test.jpg"
    save_path = "test"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load("sub_pixel_convolution.pth").to(device)
    model.eval()

    hr_image = Image.open(test_image_path).convert('RGB')
    image_width = (hr_image.width // upscale_factor) * upscale_factor
    image_height = (hr_image.height // upscale_factor) * upscale_factor
    hr_image = hr_image.resize((image_width, image_height), resample=Image.BICUBIC)
    lr_image = hr_image.resize((image_width // upscale_factor, image_height // upscale_factor), resample=Image.BICUBIC)
    lr_y, lr_cb, lr_cr = lr_image.convert('YCbCr').split()
    hr_y, _, _ = hr_image.convert('YCbCr').split()
    hr_cb = lr_cb.resize((image_width, image_height), resample=Image.BICUBIC)
    hr_cr = lr_cr.resize((image_width, image_height), resample=Image.BICUBIC)
    x = Variable(ToTensor()(lr_y)).to(device).unsqueeze(0)
    hr_y = Variable(ToTensor()(hr_y)).to(device).unsqueeze(0)
    bicubic = lr_image.resize((image_width, image_height), resample=Image.BICUBIC)

    psnr = utils.calaculate_psnr(hr_y, Variable(ToTensor()(bicubic)).to(device).unsqueeze(0))

    print('bicubic PSNR: {}'.format(psnr))
    bicubic.save(test_image_path.replace(".", "_bicubic_x{}.".format(upscale_factor)))

    with torch.no_grad():
        out = model(x).clip(0, 1).squeeze()
    print(hr_y.size())
    print(out.size())
    psnr = utils.calaculate_psnr(hr_y, out)
    print('espcn PSNR: {}'.format(psnr))
    out = utils.tensor_to_image(out)
    out_img = Image.merge('YCbCr', [out, hr_cb, hr_cr]).convert('RGB')
    out_img.save(test_image_path.replace('.', '_espcn_x{}.'.format(upscale_factor)))
