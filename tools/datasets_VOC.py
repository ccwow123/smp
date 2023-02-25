# -*- coding: utf-8 -*-
import os

from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
from torch.utils.data import Dataset as BaseDataset

class Dataset_Train(BaseDataset):
    """CamVid数据集。进行图像读取，图像增强增强和图像预处理.

    Args:
        images_dir (str): 图像文件夹所在路径
        masks_dir (str): 图像分割的标签图像所在路径
        class_values (list): 用于图像分割的所有类别数
        augmentation (albumentations.Compose): 数据传输管道
        preprocessing (albumentations.Compose): 数据预处理
    """
    # CamVid数据集中用于图像分割的所有标签类别 直接在train中导入，这里不用写了
    # CLASSES = ['background','end_skew']

    # CLASSES = ['sky', 'building']

    def __init__(
            self,
            images_dir,
            masks_dir,
            images_size=(512, 512),
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_size = images_size
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = list(range(len(classes)))
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        # image = resize_image(image, self.images_size)
        image = cv2.resize(image, self.images_size)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = cv2.resize(mask, self.images_size)
        # mask = resize_image(mask, self.images_size)

        # 从标签中提取特定的类别 (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # 图像增强应用
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # 图像预处理应用
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # print(mask.shape)
        return image, mask

    def __len__(self):
        return len(self.ids)

class Dataset_Val(BaseDataset):
    """CamVid数据集。进行图像读取，图像增强增强和图像预处理.

    Args:
        images_dir (str): 图像文件夹所在路径
        masks_dir (str): 图像分割的标签图像所在路径
        class_values (list): 用于图像分割的所有类别数
        augmentation (albumentations.Compose): 数据传输管道
        preprocessing (albumentations.Compose): 数据预处理
    """
    # CamVid数据集中用于图像分割的所有标签类别
    CLASSES = ['sky', 'building']
    # CLASSES = ['background','end_skew']
    def __init__(
            self,
            images_dir,
            # masks_dir,
            images_size=None,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.images_size = images_size

        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = list(range(len(classes)))
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        if self.images_size is not None:
            image = cv2.resize(image, self.images_size)# 改变图片分辨率

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

def resize_image(image, size=(512, 512)):
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.LANCZOS)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    # pil转cv2
    new_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)

    return new_image
