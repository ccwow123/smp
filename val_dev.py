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
from tools.mytools import *
import yaml


class Valer():
    def __init__(self, args):
        self.cfg = self._load_cfg()
        # 数据集预处理
        self.preprocessing_fn = self.get_preprocessing_fn()
        self.args = args
        # self.model_name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 工具类
        self.log_dir = self._create_folder()
        self.time_calculater = Time_calculater()
     # 创建log文件夹
    def _create_folder(self):
        log_dir = self.args.pretrained.split('/')[:-1]
        log_dir = '/'.join(log_dir)
        self.results_file = log_dir + "/val_results.txt"
        return log_dir
    def _load_cfg(self):
        with open(args.model, 'r', encoding='utf-8') as f:
            yamlresult = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.encoder = yamlresult['encoder']
        self.encoder_weights = yamlresult['encoder_weights']
        self.classes = yamlresult['classes']
        self.activation = yamlresult['activation']
        self.model_name = yamlresult['model_name']

        return yamlresult
    def _create_model(self):
        model = torch.load(self.args.pretrained)
        return model
    # 加载数据集
    def load_data(self):
        # 验证集
        x_valid_dir = os.path.join(self.args.data_path, 'val')
        y_valid_dir = os.path.join(self.args.data_path, 'valannot')
        valid_dataset = Dataset_Train(
            x_valid_dir,
            y_valid_dir,
            images_size=self.args.base_size,
            augmentation=get_validation_augmentation(base_size=self.args.base_size),
            preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.classes,
        )

        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

        return valid_loader
        # 数据集预处理
    def get_preprocessing_fn(self):
        if self.encoder_weights is not None:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        else:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, 'imagenet')

        return preprocessing_fn
    # 建立优化器和损失函数
    def create_optimizer(self,model):
        loss = losses.DiceLoss()
        return loss
    def create_metrics(self):
        metrics_ = [metrics.IoU(threshold=0.5), metrics.Fscore(beta=1, threshold=0.5),
                   metrics.Accuracy(threshold=0.5), metrics.Recall(), metrics.Precision()]
        return metrics_
    def valid_one_epoch(self,model,loss,metrics_):
        valid_epoch = train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics_,
            device=self.device,
            verbose=True,
            num_classes=len(self.classes),
        )
        return valid_epoch
    # 保存训练过程中的信息
    def save_logs(self,i, log_val, confmat):
        val_info = str(confmat)
        print(val_info)
        with open(self.results_file, "a") as f:
            f.write("Epoch: {}  \n".format(i))
            f.write("Valid: {} \n".format(log_val))
            f.write("Confusion matrix:\n {}  ".format(val_info))

            f.write("\nModel cfg: {}  \n".format(self.cfg))
            f.write("datasets: {}  \n".format(self.args.data_path))

    def run(self):
        # 创建训练集和验证集的数据加载器
        valid_loader = self.load_data()
        # 创建模型
        model = self._create_model()
        # 创建优化器和损失函数
        loss=self.create_optimizer(model)
        # 创建评价指标
        metrics_=self.create_metrics()
        # 创建一个简单的循环，用于迭代数据样本
        valid_epoch2 = self.valid_one_epoch(model,loss,metrics_)

        start_time = time.time()
        # 验证模型
        log_val, confmat = valid_epoch2.run(valid_loader)
        # 保存训练过程中的信息
        self.save_logs(1, log_val, confmat)
        # 计算训练时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training total_time: {}".format(total_time_str))


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch segnets training")
    # 主要
    # parser.add_argument('--model_name', default='unet', type=str, help='模型名称')
    parser.add_argument("--model", default= r'cfg/unet/ResNet/unet_cap_multi_res34.yaml',
                        type=str, help="选择模型,查看cfg文件夹")
    parser.add_argument("--data-path", default=r'data/E skew', help="VOCdevkit 路径")
    parser.add_argument("--base-size", default=[512, 512], type=int, help="图片缩放大小")
    parser.add_argument("--pretrained", default=r"logs/03-15 21_26_16-unet/best_model.pth", type=str, help="权重位置的路径")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    val=Valer(args)
    val.run()