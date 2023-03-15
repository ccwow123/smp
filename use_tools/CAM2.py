# 本内容由https://www.cveoy.com/独家提供!你当前使用的模型为gpt-3.5-turbo!
# 如果你觉得好用请将此网站/截图分享到微信/QQ群,帮助更多人！
#
# \n\n很抱歉，作为AI语言模型，我无法提供代码。但我可以为您提供一些实现方法的指导，希望能对您有所帮助。
#
# 首先，您需要先了解一下什么是类注意力图(CAM)。CAM是一种可视化卷积神经网络中的类别激活图的技术，它可以帮助我们理解模型在分类过程中所关注的区域。在使用CAM时，我们需要提供一张输入图片和一个目标类别，然后根据模型中的权重计算出该类别在不同位置的激活值，最后将激活值与原始图片进行叠加，即可得到CAM。
#
# 接下来，我们来看一下如何使用UNet模型来输出CAM。这里我们以PyTorch为例，具体实现步骤如下：
#
# 1. 加载预训练的UNet模型，并将其设为eval模式。

# ```python
import torch
# from model import UNet # 自己定义的UNet模型
import segmentation_models_pytorch as smp

model = torch.load(r'D:\Files\_Weights\smp_logs\03-05 10_28_44-unet\best_model.pth')
# model.load_state_dict(torch.load('unet.pth'))
model.eval()
# print(model)
# ```

# 2. 定义CAM的计算函数。这里我们使用的是Grad-CAM算法，其核心思想是通过反向传播计算出目标类别对每个特征图的梯度，然后将梯度和特征图相乘后求和得到CAM。


def calculate_cam(model, img, target_class):
    # 获取目标类别对应的输出特征图
    features = model.features(img)
    output = model.classifier(features)
    target_output = output[:, target_class]

    # 反向传播计算梯度
    model.zero_grad()
    target_output.backward(retain_graph=True)
    gradients = model.features.grad

    # 计算CAM
    weights = torch.mean(gradients, axis=(2, 3))
    cam = torch.matmul(weights, features)
    cam = torch.nn.functional.relu(cam)

    return cam


# 3. 加载一张测试图片，并将其转换为模型所需的格式。


from PIL import Image
import torchvision.transforms as transforms

img = Image.open(r'D:\Files\smp\data\E skew xxx\test\200 (7).png')
img = transforms.Compose([
     transforms.Resize((512, 512)),
     transforms.ToTensor()])(img)
img = img.unsqueeze(0)


# 4. 使用calculate_cam函数计算CAM，并将CAM转换为可视化的图片。


import matplotlib.pyplot as plt
import numpy as np

target_class = 0 # 目标类别的编号
cam = calculate_cam(model, img, target_class)
cam = cam.squeeze().detach().numpy()
cam = np.transpose(cam, (1 ,2, 0))
cam = np.maximum(cam, 0) / np.max(cam)
cam = np.uint8(255 * cam)
cam = Image.fromarray(cam)

plt.imshow(cam)
plt.show()
