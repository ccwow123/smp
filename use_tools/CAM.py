# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image


def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape  # 1,960,7,7
    output_cam = []
    for idx in class_idx:  # 只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h * w))  # [960,7*7]
        cam = weight_softmax[idx].dot(
            feature_conv.reshape((nc, h * w)))  # (5, 960) * (960, 7*7) -> (5, 7*7) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
        cam_img = np.uint8(255 * cam_img)  # Format as CV_8UC1 (as applyColorMap required)

        # output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size
        output_cam.append(cam_img)
    return output_cam





if __name__ == '__main__':
    # 加载自己的网络
    # from model import model
    model = torch.load(r'D:\Files\_Weights\smp_logs\03-05 10_28_44-unet\best_model.pth')
    class_num = 2
    # print(model)
    model_ft = model
    # model_ft.load_state_dict(torch.load('pretrain.pth', map_location=lambda storage, loc: storage))

    model_features = nn.Sequential(*list(model_ft.children())[:-2])
    print(model_ft.state_dict().keys())
    fc_weights = model_ft.state_dict()['segmentation_head.0.weight'].cpu().numpy()  # numpy数组取维度fc_weights[0].shape->(5,960)
    class_ = {0: '_background_', 1: 'abnormal'}
    model_ft.eval()
    model_features.eval()

    # 设置一个输入图片的预处理方式
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(512),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(512),
                                   transforms.CenterCrop(512),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    img_path = r'D:\Files\smp\data\E skew xxx\test\200 (7).png'  # 单张测试
    _, img_name = os.path.split(img_path)
    features_blobs = []
    img = Image.open(img_path).convert('RGB')
    img_tensor = data_transform['val'](img).unsqueeze(0)  # [1,3,224,224]
    features = model_features(img_tensor).detach().cpu().numpy()  # [1,960,7,7]
    # print(features.shape)
    logit = model_ft(img_tensor)  # [1,2] -> [ 3.3207, -2.9495]
    h_x = torch.nn.functional.softmax(logit, dim=1).data.squeeze()  # tensor([0.9981, 0.0019])

    probs, idx = h_x.sort(0, True)  # 按概率从大到小排列
    probs = probs.cpu().numpy()  # if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
    idx = idx.cpu().numpy()  # [1, 0]
    # print(probs.shape)
    # print(class_)
    for i in range(class_num):
        print('{:.3f} -> {}'.format(probs[i], class_[idx[i]]))  # 打印预测结果

    # 生成CAM
    # CAMs = returnCAM(features, fc_weights, [idx[0]])  #输出预测概率最大的特征图集对应的CAM
    CAMs = returnCAM(features, fc_weights, idx)  # 输出预测概率最大的特征图集对应的CAM
    print(img_name + ' output for the top1 prediction: %s' % class_[idx[0]])

    img = cv2.imread(img_path)
    height, width, _ = img.shape  # 读取输入图片的尺寸
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  # CAM resize match input image size
    result = heatmap * 0.3 + img * 0.5  # 比例可以自己调节

    text = '%s %.2f%%' % (class_[idx[0]], probs[0] * 100)  # 激活图结果上的文字显示
    cv2.putText(result, text, (210, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
    CAM_RESULT_PATH = r'out/heatmap/'  # CAM结果的存储地址
    if not os.path.exists(CAM_RESULT_PATH):
        os.mkdir(CAM_RESULT_PATH)
    image_name_ = img_name.split(".")[-2]
    cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_vit' + class_[idx[0]] + '.jpg', result)  # 写入存储磁盘