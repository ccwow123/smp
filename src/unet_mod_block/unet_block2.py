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

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            conv_block(in_channels, out_channels)
        )
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

        filters = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = Down(filters[0], filters[1])
        self.Conv3 = Down(filters[1], filters[2])
        self.Conv4 = Down(filters[2], filters[3])
        self.Conv5 = Down(filters[3], filters[4])

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
        x_1 = self.Conv1(x)
        x_2 = self.Conv2(x_1)
        x_3 = self.Conv3(x_2)
        x_4 = self.Conv4(x_3)
        x_5 = self.Conv5(x_4)

        y_4 = self.Up5(x_5,x_4)
        y_3 = self.Up4(y_4,x_3)
        y_2 = self.Up3(y_3,x_2)
        y_1 = self.Up2(y_2,x_1)

        out = self.segmentation_head(y_1)
        return out


if __name__ == "__main__":
    from thop import profile
    def calculater_1(model, input_size=(3, 512, 512)):
        # model = torchvision.models.alexnet(pretrained=False)
        # dummy_input = torch.randn(1, 3, 224, 224)
        dummy_input = torch.randn(1, *input_size).cuda()
        flops, params = profile(model, (dummy_input,))
        print('flops: %.2fG' % (flops / 1e9))
        print('params: %.2fM' % (params / 1e6))
        return flops / 1e9, params / 1e6
    model = MyUnet().cuda()
    input = torch.randn(1, 3, 64, 64).cuda()
    out = model(input)
    print(out.shape)
    # model = ResUNet().cuda()
    # summary(model,(3,256,256))  # 输出网络结构
    # calculater_1(model,(3,256,256))
