from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from segmentation_models_pytorch.base import SegmentationHead
from tools.mytools import model_test
from src.unet_mod_block.ResNeSt import Bottleneck
from src.unet_mod_block.mobile import Bneck
from src.unet_mod_block.shuffle import ShuffleUnit

# ----------------#
# smp-初始化函数
# ----------------#
def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
#----------------#
# resnet_block
#----------------#
class BasicBlock(nn.Module):
    expansion = 1
    '''
    expansion通道扩充比例
    out_channels就是输出的channel
    '''

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


#----------------#
#     Unet
#----------------#
class MyUnet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=2,
                 base_c: int = 32,
                 block_type='unet',
                 activation='sigmoid'):
        super().__init__()
        #          32, 64, 128, 256, 512
        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]
        # 编码器
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        if block_type=='unet':
            self.Conv2 = conv_block(filters[0], filters[1])
            self.Conv3 = conv_block(filters[1], filters[2])
            self.Conv4 = conv_block(filters[2], filters[3])
            self.Conv5 = conv_block(filters[3], filters[4])
        elif block_type=='resnet':
            self.Conv2 = BasicBlock(filters[0], filters[1])
            self.Conv3 = BasicBlock(filters[1], filters[2])
            self.Conv4 = BasicBlock(filters[2], filters[3])
            self.Conv5 = BasicBlock(filters[3], filters[4])
        elif block_type == 'resnest':
            self.Conv2 = Bottleneck(filters[0], filters[1])
            self.Conv3 = Bottleneck(filters[1], filters[2])
            self.Conv4 = Bottleneck(filters[2], filters[3])
            self.Conv5 = Bottleneck(filters[3], filters[4])
        elif block_type == 'mobile':
            self.Conv2 = nn.Sequential(Bneck(filters[0], operator_kernel=3,exp_size=filters[0],out_size=filters[0],NL='HS',s=1,SE=True),
                                        Bneck(filters[0], operator_kernel=3,exp_size=filters[0]*4,out_size=filters[1],NL='HS',s=1,SE=True),
                                        Bneck(filters[1], operator_kernel=3,exp_size=filters[0]*8,out_size=filters[1],NL='HS',s=1,SE=True))
            self.Conv3 = nn.Sequential(Bneck(filters[1], operator_kernel=3,exp_size=filters[1],out_size=filters[1],NL='HS',s=1,SE=True),
                                        Bneck(filters[1], operator_kernel=3,exp_size=filters[1]*4,out_size=filters[2],NL='HS',s=1,SE=True),
                                        Bneck(filters[2], operator_kernel=3,exp_size=filters[1]*8,out_size=filters[2],NL='HS',s=1,SE=True))
            self.Conv4 = nn.Sequential(Bneck(filters[2], operator_kernel=3,exp_size=filters[2],out_size=filters[2],NL='HS',s=1,SE=True),
                                        Bneck(filters[2], operator_kernel=3,exp_size=filters[2]*4,out_size=filters[3],NL='HS',s=1,SE=True),
                                        Bneck(filters[3], operator_kernel=3,exp_size=filters[2]*8,out_size=filters[3],NL='HS',s=1,SE=True))
            self.Conv5 = nn.Sequential(Bneck(filters[3], operator_kernel=3,exp_size=filters[3],out_size=filters[3],NL='HS',s=1,SE=True),
                                        Bneck(filters[3], operator_kernel=3,exp_size=filters[3]*4,out_size=filters[4],NL='HS',s=1,SE=True),
                                        Bneck(filters[4], operator_kernel=3,exp_size=filters[3]*8,out_size=filters[4],NL='HS',s=1,SE=True))
            # self.Conv2 = Bneck(filters[0], operator_kernel=3,exp_size=filters[0]*4,out_size=filters[1],NL='HS',s=1,SE=True)
            # self.Conv3 = Bneck(filters[1], operator_kernel=3,exp_size=filters[1]*4,out_size=filters[2],NL='HS',s=1,SE=True)
            # self.Conv4 = Bneck(filters[2], operator_kernel=3,exp_size=filters[2]*4,out_size=filters[3],NL='HS',s=1,SE=True)
            # self.Conv5 = Bneck(filters[3], operator_kernel=3,exp_size=filters[3]*4,out_size=filters[4],NL='HS',s=1,SE=True)
        elif block_type == 'shuffle':
            self.Maxpool = nn.Sequential()
            self.Conv2 = ShuffleUnit(filters[0], filters[1], 2)
            self.Conv3 = ShuffleUnit(filters[1], filters[2], 2)
            self.Conv4 = ShuffleUnit(filters[2], filters[3], 2)
            self.Conv5 = ShuffleUnit(filters[3], filters[4], 2)
        else:
            raise NotImplementedError('block_type 不存在')
        # 解码器
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])
        # self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        # self.active = torch.nn.Sigmoid()
        # ----------------#
        # smp-分割头
        # ----------------#
        self.segmentation_head = SegmentationHead(
            in_channels=filters[0],
            out_channels=out_ch,
            activation=activation,
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
    def forward(self, x):
        x_1 = self.Conv1(x)  #32，256，256
        x_2 = self.Conv2(self.Maxpool(x_1)) #64，128，128
        x_3 = self.Conv3(self.Maxpool(x_2))#128，64，64
        x_4 = self.Conv4(self.Maxpool(x_3))#256，32，32
        x_5 = self.Conv5(self.Maxpool(x_4))#512，16，16

        y_4 = self.Up5(x_5,x_4)#256，32，32
        y_3 = self.Up4(y_4,x_3)#128，64，64
        y_2 = self.Up3(y_3,x_2)#64，128，128
        y_1 = self.Up2(y_2,x_1)#32，256，256

        out = self.segmentation_head(y_1)
        return out

#----------------#
#     yolo_Unet
#----------------#
from src.unet_mod_block.yolo_block import Conv,C3,SPPF,C3TR,C3SPP,C3Ghost
class up_C3(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch,Conv_b):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv = Conv_b(in_ch, out_ch)

    def forward(self, x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class up(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        x1 = self.up(x1)
        return x1

class yolo_Unetv2(nn.Module):
    def __init__(self, in_ch=3, out_ch=2,
                 base_c: int = 64,
                 block_type='C3',
                 activation='sigmoid'):
        super().__init__()
        #           64, 128, 256, 512, 1024
        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]
        if block_type == 'C3':
            Conv_b = C3
        elif block_type == 'C3TR':
            Conv_b = C3TR
        elif block_type == 'C3SPP':
            Conv_b = C3SPP
        elif block_type == 'C3Ghost':
            Conv_b = C3Ghost
        else:
            raise NotImplementedError(f'Block type {block_type} is not implemented')
        # 编码器
        self.Conv1 =Conv(in_ch, filters[0], 6, 2, 2)
        self.Conv2 =nn.Sequential(Conv(filters[0], filters[1], 3, 2, 1),
                                  Conv_b(filters[1], filters[1]))
        self.Conv3 =nn.Sequential(Conv(filters[1], filters[2], 3, 2, 1),
                                    Conv_b(filters[2], filters[2]))
        self.Conv4 =nn.Sequential(Conv(filters[2], filters[3], 3, 2, 1),
                                    Conv_b(filters[3], filters[3]))
        self.Conv5 =nn.Sequential(Conv(filters[3], filters[4], 3, 2, 1),
                                    Conv_b(filters[4], filters[4]))
        self.SPP = SPPF(filters[4], filters[4])
        # 解码器
        self.Up4 = up_C3(filters[4], filters[3],Conv_b)
        self.Up3 = up_C3(filters[3], filters[2],Conv_b)
        self.Up2 = up_C3(filters[2], filters[1],Conv_b)
        self.Up1 = up_C3(filters[1], filters[0],Conv_b)
        self.Up0 = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(filters[0]),
                        nn.ReLU(inplace=True)
                    )

        # ----------------#
        # smp-分割头
        # ----------------#
        self.segmentation_head = SegmentationHead(
            in_channels=filters[0],
            out_channels=out_ch,
            activation=activation,
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


    def forward(self, x):
        x_1 = self.Conv1(x)#128, 160, 160
        x_2 = self.Conv2(x_1)#256, 80, 80
        x_3 = self.Conv3(x_2)#512, 40, 40
        x_4 = self.Conv4(x_3)#1024, 20, 20
        # mid
        x_5 = self.Conv5(x_4)#1024, 20, 20
        x_mid = self.SPP(x_5)#1024, 20, 20

        y_4 = self.Up4(x_mid,x_4)#512, 40, 40
        y_3 = self.Up3(y_4,x_3)#256, 80, 80
        y_2 = self.Up2(y_3,x_2)#128, 160, 160
        y_1 = self.Up1(y_2,x_1)#64, 320, 320
        y_0 = self.Up0(y_1)#32, 640, 640

        out = self.segmentation_head(y_0)
        return out




# class yolo_Unetv2(nn.Module):
#     def __init__(self, in_ch=3, out_ch=2,
#                  base_c: int = 32,
#                  block_type='C3',
#                  activation='sigmoid'):
#         super().__init__()
#         #           64, 128, 256, 512, 1024
#         filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]
#         if block_type == 'C3':
#             Con_b = C3
#         elif block_type == 'C3TR':
#             Con_b = C3TR
#         elif block_type == 'C3SPP':
#             Con_b = C3SPP
#         elif block_type == 'C3Ghost':
#             Con_b = C3Ghost
#         # 编码器
#         self.Conv1 =Conv(in_ch, filters[0], 6, 2, 2)
#         self.Conv2 =nn.Sequential(Conv(filters[0], filters[1], 3, 2, 1),
#                                   Con_b(filters[1], filters[1]))
#         self.Conv3 =nn.Sequential(Conv(filters[1], filters[2], 3, 2, 1),
#                                     Con_b(filters[2], filters[2]))
#         self.Conv4 =nn.Sequential(Conv(filters[2], filters[3], 3, 2, 1),
#                                     Con_b(filters[3], filters[3]))
#         self.Conv5 =nn.Sequential(Conv(filters[3], filters[4], 3, 2, 1),
#                                     Con_b(filters[4], filters[4]))
#         self.SPP = SPPF(filters[4], filters[4])
#         # 解码器
#         self.Up4 = Up_Con(filters[4], filters[3],Con_b)
#         self.Up3 = Up_Con(filters[3], filters[2],Con_b)
#         self.Up2 = Up_Con(filters[2], filters[1],Con_b)
#         self.Up1 = Up_Con(filters[1], filters[0],Con_b)
#         self.up0 = nn.Sequential(
#                         nn.Upsample(scale_factor=2),
#                         nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True),
#                         nn.BatchNorm2d(filters[0]),
#                         nn.SiLU()
#                     )
#         # ----------------#
#         # smp-分割头
#         # ----------------#
#         self.segmentation_head = SegmentationHead(
#             in_channels=filters[0],
#             out_channels=out_ch,
#             activation=activation,
#         )
#         self.initialize()
#
#     # ----------------#
#     # smp-初始化
#     # ----------------#
#     def initialize(self):
#         def initialize_weights():
#             for m in self.modules():
#                 # 判断是否属于Conv2d
#                 if isinstance(m, nn.Conv2d):
#                     torch.nn.init.xavier_normal_(m.weight.data)
#                     # 判断是否有偏置
#                     if m.bias is not None:
#                         torch.nn.init.constant_(m.bias.data, 0.3)
#                 elif isinstance(m, nn.Linear):
#                     torch.nn.init.normal_(m.weight.data, 0.1)
#                     if m.bias is not None:
#                         torch.nn.init.zeros_(m.bias.data)
#                 elif isinstance(m, nn.BatchNorm2d):
#                     m.weight.data.fill_(1)
#                     m.bias.data.fill_(0)
#
#         initialize_weights()
#         initialize_head(self.segmentation_head)
#
#
#     def forward(self, x):
#         x_1 = self.Conv1(x)#128, 160, 160
#         x_2 = self.Conv2(x_1)#256, 80, 80
#         x_3 = self.Conv3(x_2)#512, 40, 40
#         x_4 = self.Conv4(x_3)#1024, 20, 20
#         # mid
#         x_5 = self.Conv5(x_4)#1024, 20, 20
#         x_mid = self.SPP(x_5)#1024, 20, 20
#
#         y_4 = self.Up4(x_mid,x_4)#512, 40, 40
#         y_3 = self.Up3(y_4,x_3)#256, 80, 80
#         y_2 = self.Up2(y_3,x_2)#128, 160, 160
#         y_1 = self.Up1(y_2,x_1)#64, 320, 320
#         y_0 = self.up0 (y_1)
#
#         out = self.segmentation_head(y_0)
#         return out




if __name__ == "__main__":
    model = yolo_Unetv2()
    model_test(model,(2,3,256,256),'params')
    #----256----flops/G    params/M
    # unet      16.49G      8.64M
    # resnet    16.63G      8.81M
    # resnest   15.84G      7.95M
    # mobile    13.84G      5.71M
    # mobilev2  16.53G      11.58M
    # shuffle   13.30G      4.21M
    # yolo      10.53G      22.18M
