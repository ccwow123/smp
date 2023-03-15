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
class predicter():
    def __init__(self,args):
        self.save_dir = None
        with open(args.model, 'r', encoding='utf-8') as f:
            yamlresult = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.args = args
        self.dir = args.dir
        self.model = args.model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = yamlresult['encoder']
        self.encoder_weights = yamlresult['encoder_weights']
        self.classes = yamlresult['classes']
        self.activation = yamlresult['activation']
        self.model_name = yamlresult['model_name']
        self.preprocessing_fn = self.get_preprocessing_fn()

    # 数据集预处理
    def get_preprocessing_fn(self):
        if self.encoder_weights is not None:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        else:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, 'imagenet')

        return preprocessing_fn
    def create_save_dir(self):
        # save_dir = os.path.join('out', self.model_name)
        name=self.model_name + '_' + self.encoder
        save_dir = os.path.join('out',name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir
    def create_model(self):
        # create segmentation model with pretrained encoder
        best_model = torch.load(self.args.weight)
        return best_model
    def load_data(self):
        # create test dataset
        predict_dataset = Dataset_Val(
            self.dir,
            images_size=(512, 512),
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes ,
        )
        return predict_dataset
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
        pallette = palette_init()
        self.save_dir = self.create_save_dir()
        # load best saved checkpoint
        best_model = self.create_model()
        predict_dataset = self.load_data()
        predict_dataset_vis = Dataset_Val(
            self.dir,
            classes=self.classes,
        )
        time_list = []
        for i in range(len(predict_dataset)):
            image_vis = predict_dataset_vis[i] # 可视化的图像
            image = predict_dataset[i]# 用于预测的图像
            image_name = predict_dataset.ids[i]
            height = image_vis.shape[0]
            weight = image_vis.shape[1]
            # 图片输出路径
            img_out_path = os.path.join(self.save_dir, image_name)
            # 预热
            self._warming_model(best_model, i, image)
            # 通过图像分割得到的0-1图像pr_mask
            pr_mask, t_end, t_start = self._run_one_epoch(best_model, image, time_list)
            # 打印图像分割的时间
            print(f"{image_name} predict time: {round(t_end - t_start,4)}s")
            # 原代码
            # pr_mask = (pr_mask.squeeze().cpu().numpy().round().astype(np.uint8))#二分类的
            # pr_mask = (pr_mask.squeeze().cpu().numpy())
            # pr_mask = (np.argmax(pr_mask, axis=0) * 255 / (pr_mask.shape[0])).astype(np.uint8)
            #结果处理
            prediction = pr_mask.argmax(1).squeeze(0)# [C, H, W] -> [1, C, H, W]
            pr_mask = prediction.to("cpu").numpy().astype(np.uint8)
            # 图片分割后的黑白图像转换为彩色图像
            pr_mask = self._color(pallette, pr_mask)
            # 恢复图片原来的分辨率
            pr_mask = cv2.resize(pr_mask, (weight, height), interpolation=cv2.INTER_LINEAR)

            # 不同保存模式
            if args.method == "fusion":
                # 图像融合
                dst = cv2.addWeighted(image_vis, 0.7, pr_mask, 0.3, 0)
                cv2.imwrite(img_out_path, dst)
            elif args.method == "mask":
                # 保存图像分割后的黑白结果图像
                # cv2.imwrite(img_out_path, pr_mask)
                # 保存图像分割后的彩色结果图像
                cv2.imwrite(img_out_path, pr_mask)
            elif args.method == "contours":
                # 找到预测图中缺陷轮廓信息
                pred_img=cv2.cvtColor(pr_mask, cv2.COLOR_RGB2GRAY)
                result_img = contours_process(image_vis, pred_img,self.args.label)
                cv2.imwrite(img_out_path, result_img)

        print("average time: {}s".format(round(sum(time_list) / len(time_list),4)))

    def _run_one_epoch(self, best_model, image, time_list):
        x_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
        t_start = time_synchronized()
        pr_mask = best_model.predict(x_tensor)
        t_end = time_synchronized()
        time_list.append(t_end - t_start)
        return pr_mask, t_end, t_start


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch segnets training")
    # 主要
    parser.add_argument('--dir', type=str, default=r'data/multi/data\test', help='test image dir')
    parser.add_argument('--model', type=str, default=r'cfg/unet/Transformer/unet_cap_multi_mit_b0.yaml', help='model name')
    parser.add_argument('--weight', type=str, default=r'D:\Files\_Weights\smp_logs\03-04 16_18_44-unet\best_model.pth', help='pretrained model')
    parser.add_argument("--method", default="fusion", choices=["fusion", "mask", "contours"], help="输出方式")
    # 其他
    parser.add_argument("--label", default="End skew", type=str, help="contours方式下的标签")
    args = parser.parse_args()

    return args
# ---------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    predicter = predicter(args)
    predicter.run()