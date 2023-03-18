# -*- coding: utf-8 -*-
import os

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
    flops, params = profile(model.cuda(), (dummy_input,))
    print('flops: %.2fG' % (flops / 1e9))
    print('params: %.2fM' % (params / 1e6))
# 每层参数量 + 整体参数量 + 可训参数量
def calculater_2(model, input_size=(3, 512, 512)):
    summary(model, input_size=input_size)

# 单次计算
def single(yamlpath):
    global model
    # 读取yaml文件
    # yamlpath = r'cfg/unet/Transformer/unet_cap_multi_mit_b0.yaml'
    input_size = (3, 512, 512)
    # pass
    with open(yamlpath, 'r', encoding='utf-8') as f:
        yamlresult = yaml.load(f.read(), Loader=yaml.FullLoader)
    encoder = yamlresult['encoder']
    encoder_weights = yamlresult['encoder_weights']
    classes = yamlresult['classes']
    activation = yamlresult['activation']
    # 要计算的模型
    # model = smp.Unet(
    #     encoder_name=encoder,
    #     encoder_weights=encoder_weights,
    #     classes=len(classes),
    #     activation=activation,
    # ).cuda()
    # model = UnetRes(in_channel=3, out_channel=len(classes),depth=encoder).cuda()
    model = UnetRes_DSC(in_channel=3, out_channel=len(classes),depth=encoder).cuda()
    # 输出网络计算量
    calculater_1(model, input_size)
    # calculater_2(model,input_size)




if __name__ == "__main__":
    head =['my_unet']
    path = r'../cfg'


    yaml_list = []
    for i in head:
        temp = os.path.join(path,i)
        yaml_list.append(temp)
    for i in yaml_list:
        for j in os.listdir(i):
            yamlpath = os.path.join(i,j)
            print(yamlpath)
            single(yamlpath)
            print('-----------------------')

