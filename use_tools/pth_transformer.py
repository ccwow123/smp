# -*- coding: utf-8 -*-
import os
import torch

input_path = r'D:\Files\_Weights\unet_mod\03-20 21_04_57-unet0_None'
out_path = os.path.join(input_path,'out')

os.makedirs(out_path,exist_ok=True)
model = torch.load(os.path.join(input_path,'best_model.pth'))
torch.save(model.state_dict(),os.path.join(out_path,'best_model.pth'))
print('输出网络结构到{}'.format(out_path))