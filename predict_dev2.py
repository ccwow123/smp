# -*- coding: utf-8 -*-
import argparse
import os
import time

from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
from tools.datasets_VOC import  Dataset_Val,resize_image
from tools.augmentation import *
from tools.img_process import contours_process
import imageio
import yaml
import json
from src.unet_mod import *
from src.unet_mod_block import *
# 图像分割结果的可视化展示
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
# 时间计算
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
# 调试板初始化
def palette_init():
    palette_path = r'tools/palette_utils/palette.json'
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v
    return pallette
class Detecter():
    def __init__(self,args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = self._load_cfg()
        self.save_dir = self._create_save_dir()
        self.preprocessing_fn = self.get_preprocessing_fn()
    def _load_cfg(self):
        with open(args.model, 'r', encoding='utf-8') as f:
            yamlresult = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.encoder = yamlresult['encoder']
        self.encoder_weights = yamlresult['encoder_weights']
        self.classes = yamlresult['classes']
        self.activation = yamlresult['activation']
        self.model_name = yamlresult['model_name']

        return yamlresult
    # 创建保存文件夹
    def _create_save_dir(self):
        # save_dir = os.path.join('out', self.model_name)
        name=self.model_name + '_' + self.encoder
        save_dir = os.path.join('out',name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir
    # 获取图像预处理函数
    def get_preprocessing_fn(self):
        if self.encoder_weights :
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        else:
            preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet18', 'imagenet')

        return preprocessing_fn
    # 加载训练集
    def load_data(self):
        # create test dataset
        predict_dataset = Dataset_Val(
            self.args.dir,
            images_size=(512, 512),
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes ,
        )
        predict_dataset_vis = Dataset_Val(
            self.args.dir,
            classes=self.classes,
        )
        return predict_dataset,predict_dataset_vis
    # 创建模型
    def create_model(self):
        # create segmentation model with pretrained encoder
        if self.model_name == 'MyUnet':
            model = MyUnet(3, num_classes=len(self.classes), activation=self.activation)
            best_model = self._load_pretrained_model(model)
            # best_model = torch.load(self.args.weight)
        return best_model
    def _load_pretrained_model(self, model):
        #这里看权重文件的格式，如果是字典的话就用load_state_dict，如果是模型的话就用load_model
        checkpoint = torch.load(os.path.join(self.args.weight, 'best_model.pth'))
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded pretrained model '{}'".format(self.args.weight))
        return model
    def _warming_model(self, best_model, i, image):
        if i == 0:
            height, weight = image.shape[1], image.shape[2]
            init_img = torch.zeros((1, 3, height, weight), device=self.device)
            best_model(init_img)
    def _color(self, pallette, pr_mask):
        # 图像格式处理
        mask = Image.fromarray(pr_mask)
        mask.putpalette(pallette)
        mask = mask.convert('RGB')
        mask_cv = np.array(mask)[..., ::-1]
        return mask_cv
    def run(self):
        # 调色板
        pallette = palette_init()
        # 加载数据 前者用于推理 后者用于可视化
        predict_dataset,predict_dataset_vis = self.load_data()
        # 创建模型
        model = self.create_model()
        # 模型推理
        model.cuda().eval()
        for i in range(len(predict_dataset)):
            image,image_name = predict_dataset[i],predict_dataset.ids[i]
            # 原图大小
            image_vis = predict_dataset_vis[i]
            height, weight = image_vis.shape[0], image_vis.shape[1]
            # 图片输出路径
            img_out_path = os.path.join(self.save_dir, image_name)
            # 预热
            self._warming_model(model, i, image)
            # 推理
            with torch.no_grad():
                image = torch.from_numpy(image).to(self.device).unsqueeze(0)
                start_time = time_synchronized()
                pr_mask = model(image)#[1, 2, 512, 512]
                print(f"{image_name}--推理时间：{time_synchronized() - start_time:.3f}s")
            # 后处理
            prediction = pr_mask.argmax(1).squeeze(0).to("cpu").numpy().astype(np.uint8)
            # 图片分割后的黑白图像转换为彩色图像
            prediction = self._color(pallette, prediction)
            # 恢复图片原来的分辨率
            prediction = cv2.resize(prediction, (weight, height), interpolation=cv2.INTER_LINEAR)
            print(prediction.shape)
            # 不同保存模式
            if args.method == "fusion":
                # 图像融合
                dst = cv2.addWeighted(image_vis, 0.7, prediction, 0.3, 0)
                cv2.imwrite(img_out_path, dst)
            elif args.method == "mask":
                # 保存图像分割后的黑白结果图像
                # cv2.imwrite(img_out_path, pr_mask)
                # 保存图像分割后的彩色结果图像
                cv2.imwrite(img_out_path, prediction)
            elif args.method == "contours":
                # 找到预测图中缺陷轮廓信息
                pred_img = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
                result_img = contours_process(image_vis, pred_img, self.args.label)
                cv2.imwrite(img_out_path, result_img)




def parse_args():
    parser = argparse.ArgumentParser(description="预测")
    # 主要
    parser.add_argument('--dir', type=str, default=r'data/multi/data/test', help='test image dir')
    parser.add_argument('--model', type=str, default=r'cfg/my_new_block/unet.yaml', help='model name')
    parser.add_argument('--weight', type=str, default=r'D:\hjj\smp\logs\03-28 23_13_31-MyUnet_', help='pretrained model')
    parser.add_argument("--method", default="fusion", choices=["fusion", "mask", "contours"], help="输出方式")
    # 其他
    parser.add_argument("--label", default="End skew", type=str, help="contours方式下的标签")
    args = parser.parse_args()

    return args
# ---------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    detecter = Detecter(args)
    detecter.run()