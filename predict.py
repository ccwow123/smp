# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
import imageio


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
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

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


# ---------------------------------------------------------------
if __name__ == '__main__':

    DATA_DIR = r'examples/data/CamVid/'

    x_test_dir = os.path.join(DATA_DIR, 'test')

    img_test = cv2.imread(r'examples/data/CamVid/test/0001TP_008550.png')
    height = img_test.shape[0]
    weight = img_test.shape[1]

    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['car']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda'

    # 按照权重预训练的相同方法准备数据
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # 加载最佳模型
    best_model = torch.load('./demo/best_model.pth')

    # 创建检测数据集
    predict_dataset = Dataset(
        x_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # 对检测图像进行图像分割并进行图像可视化展示
    predict_dataset_vis = Dataset(
        x_test_dir,
        classes=CLASSES,
    )

    for i in range(len(predict_dataset)):
        # 原始图像image_vis
        image_vis = predict_dataset_vis[i].astype('uint8')
        image = predict_dataset[i]

        # 通过图像分割得到的0-1图像pr_mask
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        print(pr_mask.shape)

        # 恢复图片原来的分辨率
        image_vis = cv2.resize(image_vis, (weight, height))
        pr_mask = cv2.resize(pr_mask, (weight, height))
        # 保存图像分割后的黑白结果图像
        imageio.imwrite('car_test_out.png', pr_mask)
        # 原始图像和图像分割结果的可视化展示
        visualize(
            image=image_vis,
            predicted_mask=pr_mask
        )
