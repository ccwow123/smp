# -*- coding: utf-8 -*-
import yaml
from torchsummary import summary
import segmentation_models_pytorch as smp

if __name__ == "__main__":
    # 读取yaml文件
    yamlpath='cfg/unet_cap_multi_res101.yaml'
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
    # 输出网络结构
    summary(model,(3,512,512))