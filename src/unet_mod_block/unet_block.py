# -*- coding: utf-8 -*-
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from typing import Optional, Union, List
from src.unet_mod_block.ResNeSt import Bottleneck
# ----------------#
# smp-初始化函数
# ----------------#
def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# ----------------------------------------
#              myunet原版block
# ----------------------------------------
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )
class Up_Con(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_Con, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x1 = self.conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return x

# ----------------------------------------
#                  myunet原版
# ----------------------------------------
class MyUnet(SegmentationModel):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,
                 activation='sigmoid',):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.conv0_0 = DoubleConv(in_channels, base_c)
        self.conv1_0 = Down(base_c, base_c * 2)
        self.conv2_0 = Down(base_c * 2, base_c * 4)
        self.conv3_0 = Down(base_c * 4, base_c * 8)
        self.conv4_0 = Down(base_c * 8, base_c * 16)

        self.up3_1 = Up_Con(base_c * 16, base_c * 8)
        self.conv3_1 = DoubleConv(base_c * 16, base_c * 8)
        self.up2_1 = Up_Con(base_c * 8, base_c * 4)
        self.conv2_1 = DoubleConv(base_c * 8, base_c * 4)
        self.up1_1 = Up_Con(base_c * 4, base_c * 2)
        self.conv1_1 = DoubleConv(base_c * 4, base_c * 2)
        self.up0_1 = Up_Con(base_c * 2, base_c)
        self.conv0_1 = DoubleConv(base_c * 2, base_c)

        # ----------------#
        # smp-分割头
        # ----------------#
        self.segmentation_head = SegmentationHead(
            in_channels=base_c,
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
        )
        self.initialize()
    # ----------------#
    # smp-初始化
    # ----------------#
    def initialize(self):
        def initialize_weights():
            for m in self.modules():
                # 判断是否属于Conv2d
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                    # 判断是否有偏置
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.3)
                elif isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias.data)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.fill_(0)
        initialize_weights()
        initialize_head(self.segmentation_head)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.conv0_0(x) #(3, 512, 512)->[64,512,512]
        x1 = self.conv1_0(x0) #[128,256,256]
        x2 = self.conv2_0(x1) #[256,128,128]
        x3 = self.conv3_0(x2)# [512,64,64]
        x4 = self.conv4_0(x3)#[1024,32,32]

        x3_1 = self.up3_1(x4, x3)#[1024,64,64]
        x3_1 =self.conv3_1(x3_1)#[512,64,64]
        x2_1 = self.up2_1(x3_1, x2)#[512,128,128]
        x2_1 = self.conv2_1(x2_1)# [256,128,128]
        x1_1 = self.up1_1(x2_1, x1)#[256,256,256]
        x1_1 = self.conv1_1(x1_1)# [128,256,256]
        x0_1 = self.up0_1(x1_1, x0)#[128,512,512]
        x0_1 = self.conv0_1(x0_1)#[64,512,512]


        logits = self.segmentation_head(x0_1)# [2,512,512]

        return logits
# ----------------------------------------
#               unet-block修改版
# ----------------------------------------
class MyUnet_EX(SegmentationModel):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 2,
                 base_c: int = 64,
                 block_type='unet',
                 activation='sigmoid',):
        super().__init__()
        if block_type == 'unet':

            self.conv0_0 = DoubleConv(in_channels, base_c)
            self.conv1_0 = Down(base_c, base_c * 2)
            self.conv2_0 = Down(base_c * 2, base_c * 4)
            self.conv3_0 = Down(base_c * 4, base_c * 8)
            self.conv4_0 = Down(base_c * 8, base_c * 16)

            self.up3_1 = Up_Con(base_c * 16, base_c * 8)
            self.conv3_1 = DoubleConv(base_c * 16, base_c * 8)
            self.up2_1 = Up_Con(base_c * 8, base_c * 4)
            self.conv2_1 = DoubleConv(base_c * 8, base_c * 4)
            self.up1_1 = Up_Con(base_c * 4, base_c * 2)
            self.conv1_1 = DoubleConv(base_c * 4, base_c * 2)
            self.up0_1 = Up_Con(base_c * 2, base_c)
            self.conv0_1 = DoubleConv(base_c * 2, base_c)
        elif block_type == 'resnest':
            self.conv0_0 = DoubleConv(in_channels, base_c)
            self.conv1_0 = Bottleneck(base_c, base_c * 2)
            self.conv2_0 = Bottleneck(base_c * 2, base_c * 4)
            self.conv3_0 = Bottleneck(base_c * 4, base_c * 8)
            self.conv4_0 = Bottleneck(base_c * 8, base_c * 16)

            self.up3_1 = Up_Con(base_c * 16, base_c * 8)
            self.conv3_1 = DoubleConv(base_c * 16, base_c * 8)
            self.up2_1 = Up_Con(base_c * 8, base_c * 4)
            self.conv2_1 = DoubleConv(base_c * 8, base_c * 4)
            self.up1_1 = Up_Con(base_c * 4, base_c * 2)
            self.conv1_1 = DoubleConv(base_c * 4, base_c * 2)
            self.up0_1 = Up_Con(base_c * 2, base_c)
            self.conv0_1 = DoubleConv(base_c * 2, base_c)


        # ----------------#
        # smp-分割头
        # ----------------#
        self.segmentation_head = SegmentationHead(
            in_channels=base_c,
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
        )
        self.initialize()
    # ----------------#
    # smp-初始化
    # ----------------#
    def initialize(self):
        def initialize_weights():
            for m in self.modules():
                # 判断是否属于Conv2d
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight.data)
                    # 判断是否有偏置
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias.data, 0.3)
                elif isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight.data, 0.1)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias.data)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.fill_(0)
        initialize_weights()
        initialize_head(self.segmentation_head)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.conv0_0(x) #(3, 512, 512)->[64,512,512]
        x1 = self.conv1_0(x0) #[128,256,256]
        x2 = self.conv2_0(x1) #[256,128,128]
        x3 = self.conv3_0(x2)# [512,64,64]
        x4 = self.conv4_0(x3)#[1024,32,32]

        x3_1 = self.up3_1(x4, x3)#[1024,64,64]
        x3_1 =self.conv3_1(x3_1)#[512,64,64]
        x2_1 = self.up2_1(x3_1, x2)#[512,128,128]
        x2_1 = self.conv2_1(x2_1)# [256,128,128]
        x1_1 = self.up1_1(x2_1, x1)#[256,256,256]
        x1_1 = self.conv1_1(x1_1)# [128,256,256]
        x0_1 = self.up0_1(x1_1, x0)#[128,512,512]
        x0_1 = self.conv0_1(x0_1)#[64,512,512]


        logits = self.segmentation_head(x0_1)# [2,512,512]

        return logits
if __name__ == "__main__":
    model = MyUnet_EX(block_type='resnest').cuda()
    # model = ResUNet().cuda()
    summary(model,(3,256,256))  # 输出网络结构