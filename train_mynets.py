# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import time
from torch.utils.tensorboard import SummaryWriter

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
from tools.datasets_VOC import Dataset_Train
import yaml
from train_dev import Trainer

from src.lraspp.lraspp_model import lraspp_mobilenetv3_large

class Train_mynets(Trainer):
    def __init__(self,args):
        super(Train_mynets,self).__init__(args)
        self.args = args

        with open(args.model, 'r', encoding='utf-8') as f:
            yamlresult = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dir = args.data_path
        self.encoder = yamlresult['encoder']
        self.encoder_weights = yamlresult['encoder_weights']
        self.classes = yamlresult['classes']
        self.activation = yamlresult['activation']
        self.model_name = yamlresult['model_name']

        self.model = self.create_model()
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        self.loss = losses.DiceLoss()+losses.CrossEntropyLoss()
        self.metrics = [metrics.IoU(threshold=0.5),metrics.Recall()]
        # self.metrics = [metrics.IoU(threshold=0.5),metrics.Fscore(beta=1,threshold=0.5),metrics.Accuracy(threshold=0.5)]
        self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(), lr=args.lr),])

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.num_workers = args.num_workers

    def create_model(self):
        if self.model_name=='lraspp':
            model = lraspp_mobilenetv3_large(num_classes=len(self.classes))
        if self.args.pretrained:
            model=self._load_pretrained_model(model)
        return model
    # ?????????????????????
    def _load_pretrained_model(self, model):
        checkpoint = torch.load(self.args.pretrained, map_location=self.device)
        model.load_state_dict(checkpoint)
        print("Loaded pretrained model '{}'".format(self.args.pretrained))
        return model

def parse_args():
    parser = argparse.ArgumentParser(description="pytorch segnets training")
    # ??????
    parser.add_argument("--model", default=r"cfg/unet_cap_lraspp.yaml", type=str, help="????????????,??????cfg?????????")
    parser.add_argument("--data-path", default=r'data/multi/data', help="VOCdevkit ??????")
    parser.add_argument("--batch-size", default=2, type=int,help="????????????")
    parser.add_argument("--base-size", default=[512, 512], type=int,help="??????????????????")
    parser.add_argument("--crop-size", default=[512, 512], type=int,help="??????????????????")
    parser.add_argument("--epochs", default=2, type=int, metavar="N",help="????????????")
    parser.add_argument("--num-workers", default=0, type=int, help="???????????????????????????")
    parser.add_argument('--lr', default=0.0001, type=float, help='???????????????')
    parser.add_argument("--pretrained", default=r"", type=str, help="?????????????????????")

    # ??????

    parser.add_argument('--resume', default=r"", help='????????????????????????????????????')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='??????')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='????????????',dest='weight_decay')
    parser.add_argument('--optimizer', default='SGD', type=str, choices=['SGD', 'Adam', 'AdamW'], help='?????????')
    # ??????
    parser.add_argument('--open-tb', default=False, type=bool, help='??????tensorboard??????????????????')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    train = Train_mynets(args)
    train.run()
