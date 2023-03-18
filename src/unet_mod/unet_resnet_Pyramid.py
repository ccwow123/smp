import torch
import torch.nn as nn
from use_tools.calculater import calculater_1
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
# rfb
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        '''

        Args:
            in_planes: 输入通道数，最好不要小于8
            out_planes:  输出通道数
            stride:  步长
            scale:  缩放系数
            map_reduce:  通道数缩减系数
            vision:  可视化范围
            groups:  分组卷积
        '''
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1,
                      dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
                      dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1,
                      groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
                      dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

# sppf
class SPPF(nn.Module):
    def __init__(self,in_c=3):
        super().__init__()
        self.maxpool = nn.MaxPool2d(5, 1, padding=2)
        self.conv = nn.Conv2d(in_c*4, in_c, 1, 1, 0)

    def forward(self, x):
        o1 = self.maxpool(x)
        o2 = self.maxpool(o1)
        o3 = self.maxpool(o2)
        x = torch.cat([x, o1, o2, o3], dim=1)
        out = self.conv(x)
        return out

# CBAM
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print(self.spatial_attention(out).shape)
        out = self.spatial_attention(out) * out
        return out

# SPPCSPC
class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(c_, c_, 3, 1, 1)#我加了padding=1
        self.cv4 = nn.Conv2d(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = nn.Conv2d(4 * c_, c_, 1, 1)
        self.cv6 = nn.Conv2d(c_, c_, 3, 1,1)#我加了padding=1
        self.cv7 = nn.Conv2d(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x))) #(1, 128, 512, 512) 不加padding=1的话，这里的输出是(1, 128, 510, 510)
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))#(1, 128, 512, 512)不加padding=1的话，这里的输出是(1, 128, 508, 508)
        y2 = self.cv2(x) #(1, 128, 512, 512)
        return self.cv7(torch.cat((y1, y2), dim=1))

class Unet_resnet_RFB(nn.Module):
    def __init__(self, input_channels=3,num_classes=2,depth=18):
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
        # RFB
        self.rfb0 = BasicRFB(nb_filter[0], nb_filter[0])
        self.rfb1 = BasicRFB(nb_filter[1], nb_filter[1])
        self.rfb2 = BasicRFB(nb_filter[2], nb_filter[2])
        self.rfb3 = BasicRFB(nb_filter[3], nb_filter[3])

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_1 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_1 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

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
        x0_0 = self.rfb0(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0)) #(3, 64, 512, 512) -> (3,128,256,256)
        x1_0 = self.rfb1(x1_0)
        x2_0 = self.conv2_0(self.pool(x1_0)) #(3, 128, 256, 256) -> (3,256,128,128)
        x2_0 = self.rfb2(x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0)) #(3, 256, 128, 128) -> (3,512,64,64)
        x3_0 = self.rfb3(x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0)) #(3, 512, 64, 64) -> (3,1024,32,32)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1)) #(3,1024,32,32) -> (3, 512, 64, 64)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1)) #(3, 512, 64, 64) -> (3, 256, 128, 128)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1)) #(3, 256, 128, 128) -> (3, 128, 256, 256)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1)) #(3, 128, 256, 256) -> (3, 64, 512, 512)

        output = self.final(x0_1) #(3, 64, 512, 512) -> (3, 3, 512, 512)
        return output
class Unet_resnet_SPPF(nn.Module):
    def __init__(self, input_channels=3,num_classes=2,depth=18):
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

        # SPPF
        self.sppf0 = SPPF(nb_filter[0])
        self.sppf1 = SPPF(nb_filter[1])
        self.sppf2 = SPPF(nb_filter[2])
        self.sppf3 = SPPF(nb_filter[3])

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_1 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_1 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

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
        x0_0 = self.sppf0(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0)) #(3, 64, 512, 512) -> (3,128,256,256)
        x1_0 = self.sppf1(x1_0)
        x2_0 = self.conv2_0(self.pool(x1_0)) #(3, 128, 256, 256) -> (3,256,128,128)
        x2_0 = self.sppf2(x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0)) #(3, 256, 128, 128) -> (3,512,64,64)
        x3_0 = self.sppf3(x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0)) #(3, 512, 64, 64) -> (3,1024,32,32)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1)) #(3,1024,32,32) -> (3, 512, 64, 64)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1)) #(3, 512, 64, 64) -> (3, 256, 128, 128)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1)) #(3, 256, 128, 128) -> (3, 128, 256, 256)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1)) #(3, 128, 256, 256) -> (3, 64, 512, 512)

        output = self.final(x0_1) #(3, 64, 512, 512) -> (3, 3, 512, 512)
        return output
class Unet_resnet_CBAM(nn.Module):
    def __init__(self, input_channels=3,num_classes=2,depth=18):
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

        self.cbam0 = CBAM(channel=64)
        self.cbam1 = CBAM(channel=128)
        self.cbam2 = CBAM(channel=256)
        self.cbam3 = CBAM(channel=512)

        # decoding
        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_1 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_1 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

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
        # encoding path
        x0_0 = self.conv0_0(input)
        x0_0 = self.cbam0(x0_0) + x0_0

        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0 = self.cbam1(x1_0) + x1_0

        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.cbam2(x2_0) + x2_0

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.cbam3(x3_0) + x3_0

        x4_0 = self.conv4_0(self.pool(x3_0))    # bottleneck

        # decoding
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], dim=1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], dim=1))

        output = self.final(x0_1)
        return output
class Unet_resnet_SPPCSPC(nn.Module):
    def __init__(self, input_channels=3, num_classes=2, depth=18):
        super().__init__()
        if depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock
        elif depth == 34:
            layers = [3, 4, 6, 3]
            block = BasicBlock
        elif depth == 50:
            layers = [3, 4, 6, 3]
            block = BottleNeck
        elif depth == 101:
            layers = [3, 4, 23, 3]
            block = BottleNeck
        elif depth == 152:
            layers = [3, 8, 36, 3]
            block = BottleNeck

        nb_filter = [64, 128, 256, 512, 1024]

        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.sppcspc0 = SPPCSPC(c1=64, c2=64)
        self.sppcspc1 = SPPCSPC(c1=128, c2=128)
        self.sppcspc2 = SPPCSPC(c1=256, c2=256)
        self.sppcspc3 = SPPCSPC(c1=512, c2=512)

        self.conv3_1 = VGGBlock((nb_filter[3] + nb_filter[4]) * block.expansion, nb_filter[3],
                                nb_filter[3] * block.expansion)
        self.conv2_1 = VGGBlock((nb_filter[2] + nb_filter[3]) * block.expansion, nb_filter[2],
                                nb_filter[2] * block.expansion)
        self.conv1_1 = VGGBlock((nb_filter[1] + nb_filter[2]) * block.expansion, nb_filter[1],
                                nb_filter[1] * block.expansion)
        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

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
        x0_0 = self.sppcspc0(x0_0)
        x1_0 = self.conv1_0(self.pool(x0_0))  # (3, 64, 512, 512) -> (3,128,256,256)
        x1_0 = self.sppcspc1(x1_0)
        x2_0 = self.conv2_0(self.pool(x1_0))  # (3, 128, 256, 256) -> (3,256,128,128)
        x2_0 = self.sppcspc2(x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0))  # (3, 256, 128, 128) -> (3,512,64,64)
        x3_0 = self.sppcspc3(x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0))  # (3, 512, 64, 64) -> (3,1024,32,32)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))  # (3,1024,32,32) -> (3, 512, 64, 64)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))  # (3, 512, 64, 64) -> (3, 256, 128, 128)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_1)], 1))  # (3, 256, 128, 128) -> (3, 128, 256, 256)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_1)], 1))  # (3, 128, 256, 256) -> (3, 64, 512, 512)

        output = self.final(x0_1)  # (3, 64, 512, 512) -> (3, 3, 512, 512)
        return output
if __name__ == '__main__':
    # net = Unet_resnet_RFB(num_classes=2)
    # net = Unet_resnet_SPPCSPC(num_classes=3, depth=18)
    # x = torch.rand((3, 3, 512, 512))
    # print(net.forward(x).shape)
    net = Unet_resnet_SPPCSPC(num_classes=3, depth=18)
    calculater_1(net, (3, 64, 64))