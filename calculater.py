# -*- coding: utf-8 -*-
import torch
import torchvision
from thop import profile
import yaml
from torchsummary import summary
import segmentation_models_pytorch as smp

# 整体参数量 + 计算量
def calculater_1(model, input_size=(3, 512, 512)):
    # model = torchvision.models.alexnet(pretrained=False)
    # dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = torch.randn(1, *input_size).cuda()
    flops, params = profile(model, (dummy_input,))
    print('flops: %.2fG' % (flops / 1e9))
    print('params: %.2fM' % (params / 1e6))
# 每层参数量 + 整体参数量 + 可训参数量
def calculater_2(model, input_size=(3, 512, 512)):
    summary(model, input_size=input_size)

if __name__ == "__main__":
    # 读取yaml文件
    yamlpath='cfg/unet.yaml'
    input_size = (3, 512, 512)
    # pass
    with open(yamlpath, 'r', encoding='utf-8') as f:
        yamlresult = yaml.load(f.read(), Loader=yaml.FullLoader)
    encoder = yamlresult['encoder']
    encoder_weights = yamlresult['encoder_weights']
    classes = yamlresult['classes']
    activation = yamlresult['activation']
    # 要计算的模型
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(classes),
        activation=activation,
    ).cuda()
    # 输出网络计算量
    calculater_1(model,input_size)
    # calculater_2(model,input_size)
