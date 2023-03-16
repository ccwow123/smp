import argparse
import os
import warnings

import cv2
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import *

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


def parse_args():
    parser = argparse.ArgumentParser(description="生成CAM图")
    parser.add_argument('--data_path', type=str, default=r'../data/E skew/train', help='images dir')
    parser.add_argument('--weights_path', type=str, default=r'../logs/03-15 21_26_16-unet/best_model.pth', help='模型权重')
    parser.add_argument('--out_path', type=str, default=r'../out/CAM', help='输出路径')
    parser.add_argument("--img_size", default=(512, 512), help="图片大小")
    parser.add_argument("--classes", default=['_background_', 'abnormal'], help="训练标签")
    parser.add_argument("--category", default='abnormal', help="热图目标类别")
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='CAM方法')
    args = parser.parse_args()
    return args


def main(args):
    path = args.data_path
    out_path = args.out_path
    weights_path = args.weights_path
    img_size = args.img_size
    sem_classes = args.classes
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    car_category = sem_class_to_idx[args.category]
    os.makedirs(out_path, exist_ok=True)
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    # 1.加载模型
    model = torch.load(weights_path)
    model = model.eval()
    # print(model)
    if torch.cuda.is_available():
        model = model.cuda()
    target_layers = [model.encoder.layer4]#########################################记得这里要分析网络结构并修改
    # 2.导入图片
    img_list = os.listdir(path)
    for img_name in img_list:
        image = np.array(Image.open(os.path.join(path, img_name)))
        height, width, _ = image.shape # 获取图片的高和宽
        rgb_img = np.float32(image) / 255  # 归一化
        rgb_img = cv2.resize(rgb_img, img_size)
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()


        # 3.得到模型输出
        output = model(input_tensor)
        normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()  # 获取每个像素点的概率


        # 4.获取目标的mask
        car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
        car_mask_float = np.float32(car_mask == car_category)
        # 原来是用来进行拼接的
        # car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
        # both_images = np.hstack((rgb_img, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1))) # 拼接图片,mask要上色

        # 5.总结目标的预测热图
        targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
        # 6.不同的方法生成热图
        # with GradCAM(model=model,
        #              target_layers=target_layers,
        #              use_cuda=torch.cuda.is_available()) as cam:
        with methods[args.method](model=model,
                     target_layers=target_layers,
                     use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.resize(cam_image, (width, height))
            cv2.imwrite(os.path.join(out_path, img_name), cam_image)
            print('save image: ', img_name)
    print('done')


if __name__ == '__main__':
    args = parse_args()
    main(args)