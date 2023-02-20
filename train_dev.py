# -*- coding: utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from tools.augmentation import *
from tools.datasets_VOC import Dataset

class Trainer():
    def __init__(self,DATA_DIR):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dir = DATA_DIR
        self.encoder = 'resnet34'
        self.encoder_weights = 'imagenet'
        self.classes = ['car']
        self.activation = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
        self.model = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            classes=len(self.classes),
            activation=self.activation,
        )
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        self.batch_size = 2
        self.num_workers = 0
        self.loss = losses.DiceLoss()
        self.metrics = [metrics.IoU(threshold=0.5),]
        self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=0.0001),])
        self.epochs = 10

    def dataload(self):
        # 训练集
        x_train_dir = os.path.join(self.dir, 'train')
        y_train_dir = os.path.join(self.dir, 'trainannot')

        # 验证集
        x_valid_dir = os.path.join(self.dir, 'val')
        y_valid_dir = os.path.join(self.dir, 'valannot')

        # 创建训练集和验证集
        train_dataset = Dataset(
            x_train_dir,
            y_train_dir,
            augmentation=get_training_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes,
        )

        valid_dataset = Dataset(
            x_valid_dir,
            y_valid_dir,
            augmentation=get_validation_augmentation(),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes,
        )

        # 创建训练集和验证集的数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

        return train_loader,valid_loader

    def train_one_epoch(self):
        train_epoch = train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
        )
        return train_epoch

    def valid_one_epoch(self):
        valid_epoch = train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )
        return valid_epoch

    def run(self):
        # 创建训练集和验证集的数据加载器
        train_loader,valid_loader = self.dataload()

        # 创建一个简单的循环，用于迭代数据样本
        train_epoch2 = self.train_one_epoch()
        valid_epoch2 = self.valid_one_epoch()

        max_score = 0
        for i in range(0, self.epochs):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch2.run(train_loader)
            valid_logs = valid_epoch2.run(valid_loader)
            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, './best_model_mine.pth')
                print('Model saved!')
            if i == 5:
                self.optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')




# $# 创建模型并训练
# ---------------------------------------------------------------
if __name__ == '__main__':

    # 数据集所在的目录
    DATA_DIR = r'D:\Files\segmentation_models.pytorch-0.3.2/examples/data/CamVid'
    trainer = Trainer(DATA_DIR)
    trainer.run()

    # 如果目录下不存在CamVid数据集，则克隆下载
    # if not os.path.exists(DATA_DIR):
    #     print('Loading data...')
    #     os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    #     print('Done!')

    # # 训练集
    # x_train_dir = os.path.join(DATA_DIR, 'train')
    # y_train_dir = os.path.join(DATA_DIR, 'trainannot')
    #
    # # 验证集
    # x_valid_dir = os.path.join(DATA_DIR, 'val')
    # y_valid_dir = os.path.join(DATA_DIR, 'valannot')

    # ENCODER = 'resnet34'
    # ENCODER_WEIGHTS = 'imagenet'
    # CLASSES = ['car']
    # ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    # DEVICE = 'cuda'

    # 用预训练编码器建立分割模型
    # 使用FPN模型
    # model = smp.FPN(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     classes=len(CLASSES),
    #     activation=ACTIVATION,
    # )
    # # 使用unet++模型
    # model = smp.Unet(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     classes=len(CLASSES),
    #     activation=ACTIVATION,
    # )

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    #
    # # 加载训练数据集
    # train_dataset = Dataset(
    #     x_train_dir,
    #     y_train_dir,
    #     augmentation=get_training_augmentation(),
    #     preprocessing=get_preprocessing(preprocessing_fn),
    #     classes=CLASSES,
    # )
    #
    # # 加载验证数据集
    # valid_dataset = Dataset(
    #     x_valid_dir,
    #     y_valid_dir,
    #     augmentation=get_validation_augmentation(),
    #     preprocessing=get_preprocessing(preprocessing_fn),
    #     classes=CLASSES,
    # )

    # # 需根据显卡的性能进行设置，batch_size为每次迭代中一次训练的图片数，num_workers为训练时的工作进程数，如果显卡不太行或者显存空间不够，将batch_size调低并将num_workers调为0
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # loss = losses.DiceLoss()
    # metrics = [
    #     metrics.IoU(threshold=0.5),
    # ]
    #
    # optimizer = torch.optim.Adam([
    #     dict(params=model.parameters(), lr=0.0001),
    # ])

    # # 创建一个简单的循环，用于迭代数据样本
    # train_epoch =  train.TrainEpoch(
    #     model,
    #     loss=loss,
    #     metrics=metrics,
    #     optimizer=optimizer,
    #     device=DEVICE,
    #     verbose=True,
    # )
    #
    # valid_epoch =  train.ValidEpoch(
    #     model,
    #     loss=loss,
    #     metrics=metrics,
    #     device=DEVICE,
    #     verbose=True,
    # )

    # 进行40轮次迭代的模型训练
    # max_score = 0
    #
    # for i in range(0, 40):
    #
    #     print('\nEpoch: {}'.format(i))
    #     train_logs = train_epoch.run(train_loader)
    #     valid_logs = valid_epoch.run(valid_loader)
    #
    #     # 每次迭代保存下训练最好的模型
    #     if max_score < valid_logs['iou_score']:
    #         max_score = valid_logs['iou_score']
    #         torch.save(model, './best_model.pth')
    #         print('Model saved!')
    #
    #     if i == 25:
    #         optimizer.param_groups[0]['lr'] = 1e-5
    #         print('Decrease decoder learning rate to 1e-5!')

