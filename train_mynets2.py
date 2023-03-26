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
from thop import profile
from src.unet_mod import *
# from src.unet_mod.unet_resnet import Unet_resnet
# from src.unet_mod.smp_unet_resnet import smp_Unet_resnet
# from src.unet_mod.smp_unet import UNet
# from src.unet_mod.unet_att import AttU_Net


def calculater_1(model, input_size=(3, 512, 512)):
    # model = torchvision.models.alexnet(pretrained=False)
    # dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = torch.randn(1, *input_size).cuda()
    flops, params = profile(model, (dummy_input,))
    print('flops: %.2fG' % (flops / 1e9))
    print('params: %.2fM' % (params / 1e6))
    return flops/ 1e9, params/ 1e6
class Trainer:
    def __init__(self, args):
        self.args = args
        self.cfg = self._load_cfg()
        # 数据集预处理
        self.preprocessing_fn = self.get_preprocessing_fn()
        # self.model_name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 工具类
        self.log_dir = self._create_folder()
        self.time_calculater = Time_calculater()

    def _load_cfg(self):
        with open(self.args.model, 'r', encoding='utf-8') as f:
            yamlresult = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.encoder = yamlresult['encoder']
        self.encoder_weights = yamlresult['encoder_weights']
        self.classes = yamlresult['classes']
        self.activation = yamlresult['activation']
        self.model_name = yamlresult['model_name']

        return yamlresult

    def _create_folder(self):
        # 用来保存训练以及验证过程中信息
        if not os.path.exists("logs"):
            os.mkdir("logs")
        # 创建时间+模型名文件夹
        time_str = datetime.datetime.now().strftime("%m-%d %H_%M_%S-")
        log_dir = os.path.join("logs", time_str + self.model_name+'_'+str(self.encoder))
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.results_file = log_dir + "/{}_results{}.txt".format(self.model_name, time_str)
        # 实例化tensborad
        self.tb = SummaryWriter(log_dir=log_dir)
        # 实例化wandb
        # config = {'data-path': self.args.data_path, 'batch-size': self.args.batch_size}
        # self.wandb = wandb.init(project='newproject',name='每次改一下名称', config=config, dir=log_dir)
        print("当前进行训练: {}".format(log_dir))
        return log_dir

    def _create_model(self):
        if self.model_name == 'unet':
            model = smp.Unet(encoder_name=self.encoder, encoder_weights=self.encoder_weights, classes=len(self.classes),
                             activation=self.activation)
        elif self.model_name == 'smp_unet_resnet':
            model = smp_Unet_resnet(input_channels=3, num_classes=len(self.classes),activation=self.activation)
        elif self.model_name == 'unet0':
            model = UNet(3, num_classes=len(self.classes),activation=self.activation)
        elif self.model_name == 'unet0_CBAM':
            model = UNet_attention_ex(3, num_classes=len(self.classes),activation=self.activation,method='cbam')
        elif self.model_name == 'unet0_SE':
            model = UNet_attention_ex(3, num_classes=len(self.classes),activation=self.activation,method='se')
        elif self.model_name == 'unet0_SA':
            model = UNet_attention_ex(3, num_classes=len(self.classes), activation=self.activation, method='sa')
        elif self.model_name == 'unet0_CA':
            model = UNet_attention_ex(3, num_classes=len(self.classes), activation=self.activation, method='ca')
        elif self.model_name == 'unet0_SOCA':
            model = UNet_attention_ex(3, num_classes=len(self.classes), activation=self.activation, method='soca')
        elif self.model_name == 'unet0_res':
            model = ResUNet(3, num_classes=len(self.classes),activation=self.activation)
        elif self.model_name == 'unet0_shuffle':
            model = ShuffleUNet(3, num_classes=len(self.classes),activation=self.activation)



        # 是否加载预训练模型
        if self.args.pretrained:
            model = self._load_pretrained_model(model)
        return model
    # 加载预训练模型
    def _load_pretrained_model(self, model):
        #这里看权重文件的格式，如果是字典的话就用load_state_dict，如果是模型的话就用load_model
        checkpoint = torch.load(os.path.join(self.args.pretrained, 'best_model.pth'))
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded pretrained model '{}'".format(self.args.pretrained))
        return model
    # 加载数据集
    def load_data(self):
        # 训练集
        x_train_dir = os.path.join(self.args.data_path, 'train')
        y_train_dir = os.path.join(self.args.data_path, 'trainannot')

        # 验证集
        x_valid_dir = os.path.join(self.args.data_path, 'val')
        y_valid_dir = os.path.join(self.args.data_path, 'valannot')

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
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=self.args.num_workers)

        return train_loader, valid_loader
        # 数据集预处理
    def get_preprocessing_fn(self):

        preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')

        return preprocessing_fn
    # 建立优化器和损失函数
    def create_optimizer(self,model):
        loss = losses.DiceLoss()
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=self.args.lr), ])
        return loss,optimizer
    # 建立学习率调整策略
    def create_lr_scheduler(self,optimizer):
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return lr_scheduler
    def create_metrics(self):
        metrics_ = [metrics.IoU(threshold=0.5), metrics.Fscore(beta=1, threshold=0.5)]
        return metrics_
    def train_one_epoch(self,model,loss,metrics_,optimizer):
        train_epoch = train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics_,
            optimizer=optimizer,
            device=self.device,
            verbose=True,
        )
        return train_epoch
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
    def save_logs(self,i, log_train, log_val, confmat, optimizer):
        # 使用tb保存训练过程中的信息
        self.tb.add_scalar('train_loss', log_train['dice_loss'], i)
        # self.tb.add_scalars('val_loss', {}, i)
        # 保存验证过程中的信息
        self.tb.add_scalar('val_loss', log_val['dice_loss'], i)
        self.tb.add_scalar('PA', confmat['pa'], i)
        self.tb.add_scalar('mPA', confmat['mpa'], i)
        self.tb.add_scalar('mIoU', confmat['miou'], i)
        self.tb.add_scalar('fIoU', confmat['fwiou'], i)
        # self.tb.add_scalar('g_correct', confmat.acc_global, i)
        # self.tb.add_scalar('miou', confmat.mean_iu, i)
        # 保存学习率
        self.tb.add_scalar('learning rate',optimizer.param_groups[0]['lr'], i)
        # 保存训练过程中的信息
        val_info = str(confmat)
        print(val_info)
        with open(self.results_file, "a") as f:
            f.write("Epoch: {} - \n".format(i))
            f.write("Train: {} - \n".format(log_train))
            f.write("Valid: {} - \n".format(log_val))
            f.write("Confusion matrix:\n {}  \n".format(val_info))
            if i == self.args.epochs :
                f.write("Model cfg: {}  \n".format(self.cfg))
                f.write("datasets: {}  \n".format(self.args.data_path))
                f.write('flops：{}  params:{}  \n'.format(self.model_size[0], self.model_size[1]))

    def run(self):
        # 创建训练集和验证集的数据加载器
        train_loader, valid_loader = self.load_data()
        # 创建模型
        model = self._create_model()
        # 创建优化器和损失函数
        loss,optimizer=self.create_optimizer(model)
        # 创建学习率调整策略
        lr_scheduler=self.create_lr_scheduler(optimizer)
        # 创建评价指标
        metrics_=self.create_metrics()
        # 创建一个简单的循环，用于迭代数据样本
        train_epoch2 = self.train_one_epoch(model,loss,metrics_,optimizer)
        valid_epoch2 = self.valid_one_epoch(model,loss,metrics_)
        # 训练模型
        max_score = 0
        start_time = time.time()
        optimizer.zero_grad()
        for i in range(1, self.args.epochs+1):
            print('\nEpoch: {}'.format(i))
            # 训练模型
            log_train = train_epoch2.run(train_loader)
            # 验证模型
            log_val, confmat = valid_epoch2.run(valid_loader)
            # 调整学习率
            lr_scheduler.step()

            # 保存训练过程中的信息
            if i==self.args.epochs:
                input_size=(3,self.args.base_size[0],self.args.base_size[1])
                self.model_size=calculater_1(model,input_size)
            self.save_logs(i, log_train, log_val, confmat, optimizer)
            # 保存最好的模型
            if max_score < confmat['pa']:
                max_score = confmat['pa']
                # checkpoint = {
                #     'epoch': i,
                #     'state_dict': model.state_dict(),
                #     'optimizer': optimizer.state_dict()
                # }
                # torch.save(checkpoint,os.path.join(self.log_dir,'best_model.pth'))
                torch.save(model.state_dict(),os.path.join(self.log_dir,'best_model.pth'))
                print('--Model saved!')
            # 保存网络结构
            self.tb.add_graph(model, torch.rand(1, 3, 512, 512).to(self.device))
            # 计算剩余训练时间
            self.time_calculater.time_cal(i, self.args.epochs)
        # 计算训练时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("training total_time: {}".format(total_time_str))


def parse_args(cfgpath):
    parser = argparse.ArgumentParser(description="pytorch segnets training")
    # 主要
    parser.add_argument("--model", default=cfgpath,
                        type=str, help="选择模型,查看cfg文件夹")
    parser.add_argument("--data-path", default=r'data/E skew xxx', help="VOCdevkit 路径")
    parser.add_argument("--batch-size", default=2, type=int, help="分块大小")
    parser.add_argument("--base-size", default=[64, 64], type=int, help="图片缩放大小")
    parser.add_argument("--crop-size", default=[64, 64], type=int, help="图片裁剪大小")
    parser.add_argument("--epochs", default=10, type=int, metavar="N", help="训练轮数")
    parser.add_argument("--num-workers", default=0, type=int, help="数据加载器的线程数")
    parser.add_argument('--lr', default=0.00001, type=float, help='初始学习率')
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


if __name__ == '__main__':
    cfgpath = r'cfg/my_new_unet/unet0_SE.yaml'
    # 数据集所在的目录
    args = parse_args(cfgpath)
    trainer = Trainer(args)
    trainer.run()
