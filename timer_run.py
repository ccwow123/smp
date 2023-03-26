from threading import Timer
# from train_dev2 import *
from train_mynets2 import *
import torch
from time import sleep
# 设置延时时间
def set_timer(hour=0, min=0, sec=0):
    # 小时转秒
    def hour2sec(hour):
        return hour * 60 * 60
    # 分钟转秒
    def min2sec(min):
        return min * 60
    return hour2sec(hour) + min2sec(min) + sec
# 执行单个train
def loop(cfg_path):
    torch.cuda.empty_cache()
    args = parse_args(cfg_path)
    trainer = Trainer(args)
    trainer.run()
    torch.cuda.empty_cache()
# 执行多个train
def my_job(jobs):
    for key in jobs:
        print('-'*100,'现在执行：',key,'-'*100)
        loop(jobs[key])
        sleep(5)
if __name__ == '__main__':
    path = r'cfg/my_new_unet/'
    jobs ={
        "unet0.yaml": '',
        "unet0_CA.yaml":'',
        'unet0_CBAM.yaml':'',
        'unet0_res.yaml':'',
        'unet0_SA.yaml':'',
        'unet0_SE.yaml':'',
        'unet0_shuffle.yaml':'',
    }
    for key in jobs:
        jobs[key] = path + key
    Timer(set_timer(sec=1),my_job,(jobs,)).start()


    # cfg_path1 = r'cfg/my_new_unet/unet0.yaml'
    # cfg_path2 = r"cfg/unet/MobileOne/unet_mobileone_s2.yaml"
    # print("start")
    # # timer的第一个参数是时间（s），第二个参数是函数名，第三个参数是函数的参数，以元组的形式传入
    # Timer(min2sec(1),loop,(cfg_path1,)).start()
    # # Timer(300,loop,(cfg_path2,)).start()
    # print("end")