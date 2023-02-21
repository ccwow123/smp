# -*- coding: utf-8 -*-
import argparse
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
import imageio
import yaml

# ---------------------------------------------------------------
### Dataloader

class Dataset(BaseDataset):
    """CamVid数据集。进行图像读取，图像增强增强和图像预处理.

    Args:
        images_dir (str): 图像文件夹所在路径
        masks_dir (str): 图像分割的标签图像所在路径
        class_values (list): 用于图像分割的所有类别数
        augmentation (albumentations.Compose): 数据传输管道
        preprocessing (albumentations.Compose): 数据预处理
    """
    # CamVid数据集中用于图像分割的所有标签类别
    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist', 'unlabelled']
    CLASSES = ['end_skew']
    def __init__(
            self,
            images_dir,
            # masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, (480, 384))   # 改变图片分辨率
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 图像增强应用
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # 图像预处理应用
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        return len(self.ids)

# ---------------------------------------------------------------

def get_validation_augmentation():
    """调整图像使得图片的分辨率长宽能被32整除"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """进行图像预处理操作

    Args:
        preprocessing_fn (callbale): 数据规范化的函数
            (针对每种预训练的神经网络)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


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
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
    def create_save_dir(self):
        save_dir = os.path.join('out', self.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir
    def create_model(self):
        # create segmentation model with pretrained encoder
        best_model = torch.load(self.args.weight)
        return best_model
    def load_data(self):
        # create test dataset
        predict_dataset = Dataset(
            self.dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes ,
        )
        return predict_dataset
    def run(self):
        self.save_dir = self.create_save_dir()
        # load best saved checkpoint
        best_model = self.create_model()
        predict_dataset = self.load_data()
        predict_dataset_vis = Dataset(
            self.dir,
            classes=self.classes,
        )
        time_list = []
        for i in range(len(predict_dataset)):
            image_vis = predict_dataset_vis[i].astype('uint8') # 可视化的图像
            image = predict_dataset[i]# 用于预测的图像
            image_name = predict_dataset.ids[i]
            height = image.shape[1]
            weight = image.shape[2]
            # 通过图像分割得到的0-1图像pr_mask
            x_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
            t_start = time_synchronized()
            pr_mask = best_model.predict(x_tensor)
            t_end = time_synchronized()
            time_list.append(t_end - t_start)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            # 打印图像分割的时间
            print(f"{image_name} predict time: {round(t_end - t_start,4)}s")
            # 恢复图片原来的分辨率
            image_vis = cv2.resize(image_vis, (weight, height))# 可视化的图像
            pr_mask = cv2.resize(pr_mask, (weight, height))
            # 保存图像分割后的黑白结果图像
            cv2.imwrite(os.path.join(self.save_dir, str(i) + '.png'), pr_mask * 255)

        print("average time: {}s".format(round(sum(time_list) / len(time_list),4)))
def parse_args():
    parser = argparse.ArgumentParser(description="pytorch segnets training")
    # 主要
    parser.add_argument('--dir', type=str, default=r'data/test', help='test image dir')
    parser.add_argument('--model', type=str, default=r'cfg/unet_cap.yaml', help='model name')
    parser.add_argument('--weight', type=str, default=r'logs/02-21 09_51_18-unet/best_model_mine.pth', help='pretrained model')
    args = parser.parse_args()

    return args
# ---------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    predicter = predicter(args)
    predicter.run()