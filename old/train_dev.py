# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import time

from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import *
from torch.utils.data import DataLoader

from tools.augmentation import *
from tools.datasets_VOC import Dataset_Train
from tools.mytools import Time_calculater
import yaml


class Trainer():
    def __init__(self, args):
        self.args = args

        with open(args.model, 'r', encoding='utf-8') as f:
            yamlresult = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dir = args.data_path
        self.cfg = yamlresult
        self.encoder = yamlresult['encoder']
        self.encoder_weights = yamlresult['encoder_weights']
        self.classes = yamlresult['classes']
        self.activation = yamlresult['activation']
        self.model_name = yamlresult['model_name']

        self.model = self.create_model()
        self.preprocessing_fn = self.get_preprocessing_fn()
        # self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        self.loss = losses.DiceLoss()
        # self.metrics = [metrics.IoU(threshold=0.5),metrics.Recall()]
        self.metrics = [metrics.IoU(threshold=0.5), metrics.Fscore(beta=1, threshold=0.5),
                        metrics.Accuracy(threshold=0.5)
            , metrics.Recall(), metrics.Precision()]
        self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=args.lr), ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.num_workers = args.num_workers

        self.time_calculater = Time_calculater()

    def create_model(self):
        # create segmentation model with pretrained encoder
        if self.model_name == 'unet':
            model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.encoder_weights,
                classes=len(self.classes),
                activation=self.activation,
            )
        # 是否加载预训练模型
        if self.args.pretrained:
            model = self.load_pretrained_model(model)
        return model

    # 加载预训练模型
    def load_pretrained_model(self, model):
        checkpoint = torch.load(self.args.pretrained, map_location=self.device)
        model.load_state_dict(checkpoint['Unet'])
        print("Loaded pretrained model '{}'".format(self.args.pretrained))
        return model

    def dataload(self):
        # 训练集
        x_train_dir = os.path.join(self.dir, 'train')
        y_train_dir = os.path.join(self.dir, 'trainannot')

        # 验证集
        x_valid_dir = os.path.join(self.dir, 'val')
        y_valid_dir = os.path.join(self.dir, 'valannot')

        # 创建训练集和验证集
        train_dataset = Dataset_Train(
            x_train_dir,
            y_train_dir,
            images_size=self.args.base_size,
            augmentation=get_training_augmentation(base_size=self.args.base_size, crop_size=self.args.crop_size),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes,
        )

        valid_dataset = Dataset_Train(
            x_valid_dir,
            y_valid_dir,
            images_size=self.args.base_size,
            augmentation=get_validation_augmentation(base_size=self.args.base_size),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes,
        )

        # 创建训练集和验证集的数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

        return train_loader, valid_loader

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
            num_classes=len(self.classes),
        )
        return valid_epoch

    # 创建log文件夹
    def create_folder(self):
        # 用来保存训练以及验证过程中信息
        if not os.path.exists("logs"):
            os.mkdir("logs")
        # 创建时间+模型名文件夹
        time_str = datetime.datetime.now().strftime("%m-%d %H_%M_%S-")
        log_dir = os.path.join("logs", time_str + self.model_name)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.results_file = log_dir + "/{}_results{}.txt".format(self.model_name, time_str)
        # 实例化tensborad
        self.tb = SummaryWriter(log_dir=log_dir)
        return log_dir

    # 数据集预处理
    def get_preprocessing_fn(self):
        if self.encoder_weights is not None:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        else:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, 'imagenet')

        return preprocessing_fn

    # 保存训练过程中的信息
    def save_logs(self, i, log_dir, max_score, train_logs, val_info, valid_logs, confmat):
        # 使用tb保存训练过程中的信息
        self.tb.add_scalar('train_loss', train_logs['dice_loss'], i)
        # self.tb.add_scalar('train_iou_score', train_logs['iou_score'], i)
        # self.tb.add_scalar('train_recall', train_logs['recall'], i)
        # self.tb.add_scalar('train_fscore', train_logs['fscore'], i)
        # self.tb.add_scalar('train_accuracy', train_logs['accuracy'], i)
        # self.tb.add_scalar('train_precision', train_logs['precision'], i)
        # 保存验证过程中的信息
        self.tb.add_scalar('val_loss', valid_logs['dice_loss'], i)
        # self.tb.add_scalar('val_iou_score', valid_logs['iou_score'], i)
        # self.tb.add_scalar('val_recall', valid_logs['recall'], i)
        # self.tb.add_scalar('val_fscore', valid_logs['fscore'], i)
        # self.tb.add_scalar('val_accuracy', valid_logs['accuracy'], i)
        # self.tb.add_scalar('val_precision', valid_logs['precision'], i)
        # 保存验证信息2
        self.tb.add_scalar('global correct', confmat.acc_global, i)
        self.tb.add_scalar('mean IoU', confmat.mean_iu, i)
        # 保存学习率
        self.tb.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], i)
        # 保存网络图
        if i == 0:
            self.tb.add_graph(self.model,
                              torch.rand(1, 3, self.args.crop_size[0], self.args.crop_size[1]).to(self.device))
        # 保存训练过程中的信息
        with open(self.results_file, "a") as f:
            f.write("Epoch: {} - \n".format(i))
            f.write("Train: {} - \n".format(train_logs))
            f.write("Valid: {} - \n".format(valid_logs))
            f.write("Confusion matrix:\n {} - \n".format(val_info))
            if i == self.epochs - 1:
                f.write("\n\nModel cfg: {} - \n".format(self.cfg))
                f.write("datasets: {} - \n".format(self.dir))
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(self.model, log_dir + '/best_model.pth')
            print('Model saved!')

    def run(self):
        log_dir = self.create_folder()

        # 创建训练集和验证集的数据加载器
        train_loader, valid_loader = self.dataload()

        # 创建一个简单的循环，用于迭代数据样本
        train_epoch2 = self.train_one_epoch()
        valid_epoch2 = self.valid_one_epoch()

        max_score = 0
        start_time = time.time()
        for i in range(0, self.epochs):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch2.run(train_loader)
            # 学习率调整
            self.scheduler.step()
            valid_logs, confmat = valid_epoch2.run(valid_loader)
            val_info = str(confmat)
            print(val_info)
            self.save_logs(i, log_dir, max_score, train_logs, val_info, valid_logs, confmat)
            self.time_calculater.time_cal(i, self.epochs)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training total_time: {}".format(total_time_str))


def parse_args(cfg_path):
    parser = argparse.ArgumentParser(description="pytorch segnets training")
    # 主要
    parser.add_argument("--model", default=cfg_path,
                        type=str, help="选择模型,查看cfg文件夹")
    parser.add_argument("--data-path", default=r'data/E skew xxx', help="VOCdevkit 路径")
    parser.add_argument("--batch-size", default=6, type=int, help="分块大小")
    parser.add_argument("--base-size", default=[64, 64], type=int, help="图片缩放大小")
    parser.add_argument("--crop-size", default=[64, 64], type=int, help="图片裁剪大小")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="训练轮数")
    parser.add_argument("--num-workers", default=0, type=int, help="数据加载器的线程数")
    parser.add_argument('--lr', default=0.0001, type=float, help='初始学习率')
    parser.add_argument("--pretrained", default=r"", type=str, help="权重位置的路径")

    # 暂无

    parser.add_argument('--resume', default=r"", help='继续训练的权重位置的路径')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='动量')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='权重衰减', dest='weight_decay')
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'Adam', 'AdamW'], help='优化器')
    # 其他
    parser.add_argument('--open-tb', default=False, type=bool, help='使用tensorboard保存网络结构')

    args = parser.parse_args()

    return args


# $# 创建模型并训练
# ---------------------------------------------------------------
if __name__ == '__main__':
    cfg_path = r'cfg/unet/MobileOne/unet_mobileone_s0.yaml'
    # 数据集所在的目录
    args = parse_args(cfg_path)
    # # wanDB设置
    # config = {'data-path': args.data_path,'batch-size': args.batch_size}
    # wandb.init(project="smp",
    #            name='每次改一下名称',
    #            config=config)

    trainer = Trainer(args)
    trainer.run()
