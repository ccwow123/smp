# -*- coding: utf-8 -*-
import os

from torch.utils.tensorboard import SummaryWriter
import torch

# tensorboard --logdir net_structure 查看网络结构
out_folder = "../net_structure"
pth = r"D:\Files\_Weights\smp_logs\03-06 09_54_35-unet\best_model.pth"
input = torch.randn(1, 3, 512, 512).cuda()

out=os.path.join(out_folder,pth.split('\\')[-2].split('-')[-1])
model =torch.load(pth).cuda()
os.makedirs(out, exist_ok=True)
writer = SummaryWriter(log_dir=out, comment="123")
writer.add_graph(model, input)
writer.close()
print('输出网络结构到{}'.format(out))