from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

from torchsummary import summary
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from thop import profile
from src.unet_mod_block.ResNeSt import Bottleneck
from src.unet_mod_block.mobile import Bneck
from src.unet_mod_block.shuffle import ShuffleUnit

def calculater_1(model, input_size=(3, 512, 512)):
    # model = torchvision.models.alexnet(pretrained=False)
    # dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = torch.randn(1, *input_size).cuda()
    flops, params = profile(model, (dummy_input,))
    print('flops: %.2fG' % (flops / 1e9))
    print('params: %.2fM' % (params / 1e6))
    return flops / 1e9, params / 1e6
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


if __name__ == "__main__":
    model = MyUnet(block_type='mobile').cuda()

    # 测试一，输出shape
    # input = torch.randn(2, 3, 256, 256).cuda()
    # out = model(input)
    # print(out.shape)

    # 测试二，输出网络结构
    # summary(model,(3,256,256))  # 输出网络结构

    # 测试三，计算参数量
    calculater_1(model,(3,256,256))

    #-----------flops/G    params/M
    # unet      16.49G      8.64M
    # resnet    16.63G      8.81M
    # resnest   15.84G      7.95M
    # mobile    13.84G      5.71M
    # mobilev2  16.53G      11.58M
    # shuffle   13.30G      4.21M