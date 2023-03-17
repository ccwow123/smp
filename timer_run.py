from threading import Timer
# from train_dev2 import *
from train_mynets2 import *


# 小时转秒
def hour2sec(hour):
    return hour * 60 * 60
# 分钟转秒
def min2sec(min):
    return min * 60
def loop(cfg_path):
    args = parse_args(cfg_path)
    trainer = Trainer(args)
    trainer.run()
if __name__ == '__main__':
    cfg_path1 = r'cfg/my_unet/unet_mydense.yaml'
    # cfg_path2 = r"cfg/unet/MobileOne/unet_mobileone_s2.yaml"
    print("start")
    # timer的第一个参数是时间（s），第二个参数是函数名，第三个参数是函数的参数，以元组的形式传入
    Timer(1,loop,(cfg_path1,)).start()
    # Timer(300,loop,(cfg_path2,)).start()
    print("end")
