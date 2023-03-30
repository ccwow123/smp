# -*- coding: utf-8 -*-
import datetime
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
# import wandb
class Time_calculater(object):
    def __init__(self):
        self.start=time.time()
        self.last_time=self.start
        self.remain_time=0
    #定义将秒转换为时分秒格式的函数
    def time_change(self,time_init):
        time_list = []
        if time_init/3600 > 1:
            time_h = int(time_init/3600)
            time_m = int((time_init-time_h*3600) / 60)
            time_s = int(time_init - time_h * 3600 - time_m * 60)
            time_list.append(str(time_h))
            time_list.append('h ')
            time_list.append(str(time_m))
            time_list.append('m ')

        elif time_init/60 > 1:
            time_m = int(time_init/60)
            time_s = int(time_init - time_m * 60)
            time_list.append(str(time_m))
            time_list.append('m ')
        else:
            time_s = int(time_init)

        time_list.append(str(time_s))
        time_list.append('s')
        time_str = ''.join(time_list)
        return time_str
    def time_cal(self,i,N):
        now_time=time.time()
        self.remain_time=(now_time-self.last_time)*(N-i-1)
        self.last_time=now_time
        print("剩余时间："+self.time_change(self.remain_time))
def model_test(model,input_size,method, device='cuda'):
    '''

    Args:
        model: 要进行测试的模型
        input_size:  输入tensor的尺寸 (2,3,256,256)
        method:  测试方法，shape/summary/params
        device:  测试设备，cuda/cpu

    Returns: None

    '''
    from thop import profile
    from torchsummary import summary
    def calculater_1(model, input_size=(3, 512, 512), device='cuda'):
        # model = torchvision.models.alexnet(pretrained=False)
        # dummy_input = torch.randn(1, 3, 224, 224)
        # dummy_input = torch.randn(1, *input_size).cuda()
        dummy_input = torch.randn(1, *input_size).to(device)
        flops, params = profile(model, (dummy_input,))
        print('flops: %.2fG' % (flops / 1e9))
        print('params: %.2fM' % (params / 1e6))
        return flops / 1e9, params / 1e6

    model = model.to(device)
    if method == 'shape':
        input = torch.randn(input_size).to(device)
        out = model(input)
        print('out.shape:', out.shape)
    elif method == 'summary':
        summary(model, input_size[1:])
    elif method == 'params':
        calculater_1(model, input_size[1:], device=device)

class Train_base:
    def __init__(self,args):
        self.args = args
        # self.model_name = args.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 工具类
        self.log_dir = self._create_folder()
        self.time_calculater = Time_calculater()
        # 创建log文件夹

    def _create_folder(self):
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
        # 实例化wandb
        # config = {'data-path': self.args.data_path, 'batch-size': self.args.batch_size}
        # self.wandb = wandb.init(project='newproject',name='每次改一下名称', config=config, dir=log_dir)
        return log_dir
    # 创建模型
    def _create_model(self):
        # 创建模型
        model = None
        if self.args.pretrained:
            model = self._load_pretrained_model(model)
        return model
    # 加载预训练模型
    def _load_pretrained_model(self, model):
        #这里看权重文件的格式，如果是字典的话就用load_state_dict，如果是模型的话就用load_model
        checkpoint = torch.load(self.args.pretrained, map_location=self.device)
        model.load_state_dict(checkpoint)
        print("Loaded pretrained model '{}'".format(self.args.pretrained))
        return model
    # 加载数据集
    def load_data(self):
        pass
        # return train_loader,valid_loader
    # 建立优化器和损失函数
    def create_optimizer(self,model):
        # 损失函数
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        return criterion,optimizer
    # 建立学习率调整策略
    def create_lr_scheduler(self,optimizer):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return scheduler

    def train_one_epoch(self,*args):
        pass
    def valid_one_epoch(self,*args):
        pass
    # 保存训练过程中的信息
    def save_logs(self, i, log_train, log_val,*args):
        # 使用tb保存训练过程中的信息
        self.tb.add_scalar("train_loss", log_train, i)
        self.tb.add_scalar("valid_loss", log_val, i)
        # 使用wandb保存训练过程中的信息
        # wandb.log({"train_loss": log_train, "valid_loss": log_val})
        # 保存训练过程中的信息
        with open(self.results_file, "a") as f:
            f.write("Epoch: {} - \n".format(i))
            f.write("Train: {} - \n".format(log_train))
            f.write("Valid: {} - \n".format(log_val))

    def run(self):
        # 创建训练集和验证集的数据加载器
        train_loader, valid_loader = self.load_data()
        # 创建模型
        model = self._create_model()
        # 创建优化器和损失函数
        loss,optimizer=self.create_optimizer(model)
        # 创建学习率调整策略
        lr_scheduler=self.create_lr_scheduler(optimizer)
        # 创建一个简单的循环，用于迭代数据样本
        train_epoch = self.train_one_epoch(model,train_loader)
        val_epoch = self.valid_one_epoch(model,valid_loader)
        for i in range(0, self.epochs):
            print("Epoch {}/{}".format(i, self.epochs))
            log_train=train_epoch(train_loader)
            log_val=val_epoch(valid_loader)
            # 保存训练过程中的信息
            self.save_logs(i, log_train, log_val)
            # 保存模型
            if i % self.args.save_freq == 0:
                torch.save(self.model.state_dict(), self.log_dir + "/{}_epoch_{}.pth".format(self.model_name, i))

if __name__ == '__main__':
    time_calculater=Time_calculater()
    N=10#实际使用时用相应变量替换掉
    for i in range(N):
        time.sleep(1)#为了测试效果添加的
        time_calculater.time_cal(i,N)