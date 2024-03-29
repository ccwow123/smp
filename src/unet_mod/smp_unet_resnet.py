import torch
import torch.nn as nn
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from typing import Optional, Union, List
# ----------------#
# smp-初始化函数
# ----------------#
def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
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


class BottleNeck(nn.Module):
    expansion = 4

    '''
    espansion是通道扩充的比例
    注意实际输出channel = middle_channels * BottleNeck.expansion
    '''

    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, middle_channels * BottleNeck.expansion, stride=stride, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class VGGBlock(nn.Module):  # vgg的block作为resnet的conv
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)
        return out


class Unet_resnet(nn.Module):
    def __init__(self, input_channels=3,num_classes=2,depth=18,activation='sigmoid'):
        super().__init__()
        if depth == 18:
            layers = [2, 2, 2, 2]
            block=BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block=BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block=BottleNeck
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block=BottleNeck
        elif depth == 152:
            layers = [3, 8, 36, 3]
            block=BottleNeck

        nb_filter = [64, 128, 256, 512, 1024]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_1 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_1 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        # self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        # 分割头
        self.segmentation_head = SegmentationHead(
            in_channels=nb_filter[0],
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
        )

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        '''
        middle_channels中间维度，实际输出channels = middle_channels * block.expansion
        num_blocks，一个Layer包含block的个数
        '''

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input) #(3, 3, 512, 512) -> (3,64,512,512)
        x1_0 = self.conv1_0(self.pool(x0_0)) #(3, 64, 512, 512) -> (3,128,256,256)
        x2_0 = self.conv2_0(self.pool(x1_0)) #(3, 128, 256, 256) -> (3,256,128,128)
        x3_0 = self.conv3_0(self.pool(x2_0)) #(3, 256, 128, 128) -> (3,512,64,64)
        x4_0 = self.conv4_0(self.pool(x3_0)) #(3, 512, 64, 64) -> (3,1024,32,32)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1)) #(3,1024,32,32) -> (3, 512, 64, 64)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1)) #(3, 512, 64, 64) -> (3, 256, 128, 128)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1)) #(3, 256, 128, 128) -> (3, 128, 256, 256)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1)) #(3, 128, 256, 256) -> (3, 64, 512, 512)

        # output = self.final(x0_1) #(3, 64, 512, 512) -> (3, 3, 512, 512)
        output = self.segmentation_head(x0_1)
        return output

class smp_Unet_resnet(SegmentationModel):
    def __init__(self, input_channels=3,num_classes=2,depth=18,activation='sigmoid',aux_params: Optional[dict] = None,):
        super().__init__()
        if depth == 18:
            layers = [2, 2, 2, 2]
            block=BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block=BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block=BottleNeck
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block=BottleNeck
        elif depth == 152:
            layers = [3, 8, 36, 3]
            block=BottleNeck

        nb_filter = [64, 128, 256, 512, 1024]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_1 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_1 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        # ----------------#
        # smp-分割头
        # ----------------#
        self.segmentation_head = SegmentationHead(
            in_channels=nb_filter[0],
            out_channels=num_classes,
            activation=activation,
            kernel_size=3,
        )
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.initialize()
        self.initialize_weights()
    # ----------------#
    # smp-初始化
    # ----------------#
    def initialize(self):
        # init.initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)
    def initialize_weights(self):
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
    def _make_layer(self, block, middle_channel, num_blocks, stride):
        '''
        middle_channels中间维度，实际输出channels = middle_channels * block.expansion
        num_blocks，一个Layer包含block的个数
        '''

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)  # (3, 3, 512, 512) -> (3,64,512,512)
        x1_0 = self.conv1_0(self.pool(x0_0))  # (3, 64, 512, 512) -> (3,128,256,256)
        x2_0 = self.conv2_0(self.pool(x1_0))  # (3, 128, 256, 256) -> (3,256,128,128)
        x3_0 = self.conv3_0(self.pool(x2_0))  # (3, 256, 128, 128) -> (3,512,64,64)
        x4_0 = self.conv4_0(self.pool(x3_0))  # (3, 512, 64, 64) -> (3,1024,32,32)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  # (3,1024,32,32) -> (3, 512, 64, 64)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))  # (3, 512, 64, 64) -> (3, 256, 128, 128)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))  # (3, 256, 128, 128) -> (3, 128, 256, 256)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))  # (3, 128, 256, 256) -> (3, 64, 512, 512)

        # ----------------#
        # smp-输出
        # ----------------#
        # output = self.final(x0_1) #(3, 64, 512, 512) -> (3, 3, 512, 512)
        masks = self.segmentation_head(x0_1)

        return masks
if __name__ == '__main__':
    net = Unet_resnet(num_classes=2)
    print(net)
    x = torch.rand((3, 3, 512, 512))
    print(net.forward(x).shape)