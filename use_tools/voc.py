import os
import cv2
import numpy as np
import json
import random
import shutil
from tqdm import tqdm

'''
制作一个只包含分类标注的标签图像，假如我们分类的标签为cat和dog，那么该标签图像中，Background为0，cat为1，dog为2。
我们首先要创建一个和原图大小一致的空白图像，该图像所有像素都是0，这表示在该图像中所有的内容都是Background。
然后根据标签对应的区域使用与之对应的类别索引来填充该图像，也就是说，将cat对应的区域用1填充，dog对应的区域用2填充。
特别注意的是，一定要有Background这一项且一定要放在index为0的位置。
'''
def make_mask(dir,classes):
    os.makedirs(os.path.join(dir,'masks'), exist_ok=True)
    # 分类标签，一定要包含'Background'且必须放在最前面
    category_types = classes
    # 将图片标注json文件批量生成训练所需的标签图像png
    imgpath_list = os.listdir(os.path.join(dir,'img_data'))
    for img_path in imgpath_list:
        img_name = img_path.split('.')[0]
        img = cv2.imread(os.path.join(dir,'img_data', img_path))
        h, w = img.shape[:2]
        # 创建一个大小和原图相同的空白图像
        mask = np.zeros([h, w, 1], np.uint8)

        with open(os.path.join(dir,'data_annotated',img_name+'.json'), encoding='utf-8') as f:
            label = json.load(f)

        masks = label['shapes']
        for shape in masks:
            category = shape['label']
            points = shape['points']
            # 将图像标记填充至空白图像
            points_array = np.array(points, dtype=np.int32)
            mask = cv2.fillPoly(mask, [points_array], category_types.index(category))

        # 生成的标注图像必须为png格式
        cv2.imwrite(os.path.join(dir,'masks',img_name+'.png'), mask)
        print('mask image saved to: ', os.path.join(dir,'masks',img_name+'.png'))
    print('mask images generated successfully!')
# 之后完善可视化函数
def visualization(dir,classes):
    category_types = classes
    # 将图片标注json文件批量生成训练所需的标签图像png
    imgpath_list = os.listdir(os.path.join(dir,'img_data'))
    for img_path in imgpath_list:
        img_name = img_path.split('.')[0]
        img = cv2.imread(os.path.join(dir,'img_data', img_path))
        h, w = img.shape[:2]
        # 创建一个大小和原图相同的空白图像
        mask = np.zeros([h, w, 1], np.uint8)

        with open(os.path.join(dir,'data_annotated',img_name+'.json'), encoding='utf-8') as f:
            label = json.load(f)

        masks = label['shapes']
        for shape in masks:
            category = shape['label']
            points = shape['points']

            # 将图像标记填充至空白图像
            points_array = np.array(points, dtype=np.int32)
            # mask = cv2.fillPoly(mask, [points_array], category_types.index(category))

            if category == 'Gingerbread':
                # 调试时将某种标注的填充颜色改为255，便于查看用，实际时不需进行该操作
                mask = cv2.fillPoly(mask, [points_array], 125)
            elif category == 'Coconutmilk':
                mask = cv2.fillPoly(mask, [points_array], 255)
            else:
                mask = cv2.fillPoly(mask, [points_array], category_types.index(category))

        cv2.imshow('mask', mask)
        cv2.waitKey(0)


def split_data(dir,classes,train_percent=0.7,val_percent=0.2,test_percent=0.1):
    '''
    ├── data(按照7:2:1比例划分)
    │   ├── train 存放用于训练的图片
    │   ├── trainannot 存放用于训练的图片标注
    │   ├── val 存放用于验证的图片
    │   ├── valannot 存放用于验证的图片标注
    │   ├── test 存放用于测试的图片
    │   ├── testannot 存放用于测试的图片标注
    '''
    # 创建数据集文件夹
    dirpath_list = ['data/train', 'data/trainannot', 'data/val', 'data/valannot', 'data/test', 'data/testannot']
    for dirpath in dirpath_list:
        # if os.path.exists(dirpath):
        #     shutil.rmtree(dirpath)   # 删除原有的文件夹
        #     os.makedirs(dirpath)   # 创建文件夹
        # elif not os.path.exists(dirpath):
        #     os.makedirs(dirpath)
        os.makedirs(os.path.join(dir,dirpath), exist_ok=True)


    # # 训练集、验证集、测试集所占比例
    # train_percent = 0.7
    # val_percent = 0.2
    # test_percent = 0.1

    # 数据集原始图片所存放的文件夹，必须为png文件
    imagefilepath = os.path.join(dir,'img_data')
    total_img = os.listdir(imagefilepath)
    # 所有数据集的图片名列表
    total_name_list = [row.split('.')[0] for row in total_img]
    num = len(total_name_list)
    num_list = range(num)
    # 训练集、验证集、测试集所包含的图片数目
    train_tol = int(num * train_percent)
    val_tol = int(num * val_percent)
    test_tol = int(num * test_percent)

    # 训练集在total_name_list中的index
    train_numlist = random.sample(num_list, train_tol)
    # 验证集在total_name_list中的index
    val_test_numlist = list(set(num_list) - set(train_numlist))
    val_numlist = random.sample(val_test_numlist, val_tol)
    # 测试集在total_name_list中的index
    test_numlist = list(set(val_test_numlist) - set(val_numlist))

    # 将数据集和标签图片安装分类情况依次复制到对应的文件夹
    for i in tqdm(train_numlist,desc='train'):
        img_path = os.path.join(dir,'img_data',total_name_list[i]+'.jpg')
        new_path = os.path.join(dir,'data/train',total_name_list[i]+'.png')
        shutil.copy(img_path, new_path)
        img_path = os.path.join(dir,'masks',total_name_list[i]+'.png')
        new_path = os.path.join(dir,'data/trainannot',total_name_list[i]+'.png')
        shutil.copy(img_path, new_path)
    for i in tqdm(val_numlist,desc='val'):
        img_path = os.path.join(dir,'img_data',total_name_list[i]+'.jpg')
        new_path = os.path.join(dir,'data/val',total_name_list[i]+'.png')
        shutil.copy(img_path, new_path)
        img_path = os.path.join(dir,'masks',total_name_list[i]+'.png')
        new_path = os.path.join(dir,'data/valannot',total_name_list[i]+'.png')
        shutil.copy(img_path, new_path)
    for i in tqdm(test_numlist,desc='test'):
        img_path = os.path.join(dir,'img_data',total_name_list[i]+'.jpg')
        new_path = os.path.join(dir,'data/test',total_name_list[i]+'.png')
        shutil.copy(img_path, new_path)
        img_path = os.path.join(dir,'masks',total_name_list[i]+'.png')
        new_path = os.path.join(dir,'data/testannot',total_name_list[i]+'.png')
        shutil.copy(img_path, new_path)
    print('数据集划分完成！')


# 生成单类别的mask
def one_class():
    dir = r'D:\Files\_datasets\VOC_Seg\wait_to_process'
    classes = ['_background_', 'extension']
    dir= os.path.join(dir,classes[1])
    # 创建mask
    make_mask(dir,classes)
    # 划分数据集
    split_data(dir,classes,train_percent=0.7,val_percent=0.2,test_percent=0.1)
# 生成多类别的mask
def multi_class():
    dir = r'D:\Files\_datasets\VOC_Seg\wait_to_process'
    classes = ['_background_', 'E_collapse_angle','E_skew','E_exposure','P_extend','P_broken']
    for i in classes:
        cls2=['_background_']
        cls2.append(i)
        if i == '_background_':
            continue
        # 获取当前类别的文件夹
        path = os.path.join(dir,i.replace('_', ' '))
        print(f'当前文件夹:{path}')
        print(f'当前类别:{cls2}')
        # 创建mask
        make_mask(path,cls2)
        # 划分数据集
        split_data(path,cls2,train_percent=0.7,val_percent=0.2,test_percent=0.1)

if __name__ == '__main__':
    # one_class()
    # multi_class()

    dir = r'D:\Files\_datasets\VOC_Seg\wait_to_process'
    classes = ['_background_', 'collapse']
    dir= os.path.join(dir,'E collapse angle')
    # 创建mask
    make_mask(dir,classes)
    # visualization(dir,classes)
    # 划分数据集
    split_data(dir,classes,train_percent=0.7,val_percent=0.2,test_percent=0.1)