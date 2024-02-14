import glob
from PIL import Image
from ultralytics import YOLO
import csv
import os
from os.path import join , basename
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# 模型路径
model = YOLO(r'runs/segment/train14/weights/best.pt')
# 图片路径
source = 'image'
# 预测图片的保存目录
pred_dir = r'predictOutputDir/infers/infers'


# 如果保存的话：
results = model(source=source,save=True, name='./Pre_Dir',show_labels=False,show_conf=False,boxes=False)

# 如果不保存的话：
# results = model(source=source,show_labels=False,show_conf=False,boxes=False)

for result in results:
    image_name = basename(result.path)  # 提取图片名称
    mask_name = f"{os.path.splitext(image_name)[0]}.png"  # 根据图片名称生成保存结果的名称
    pred_image_path = join(pred_dir, mask_name)# 图片保存路径
   
    # 检测到时：
    # if result.masks is not None and len(result.masks) > 0:
    if True:
        masks_data = result.masks.data
        for index, mask in enumerate(masks_data):
            mask = mask.cpu().numpy() * 255
            # cv2.imwrite(f'./output_{index}.png', mask)
            cv2.imwrite(pred_image_path , mask)


import os  
from PIL import Image  
  
def convert_to_1bit(folder_path):  
    # 遍历文件夹中的所有图像文件  
    for filename in os.listdir(folder_path):  
        if filename.endswith(".png") or filename.endswith(".jpg"):  # 仅处理PNG和JPG格式的图像  
            file_path = os.path.join(folder_path, filename)  
              
            # 打开图像  
            img = Image.open(file_path)  
              
            # 确保图像是灰度的  
            if img.mode != 'L':  
                img = img.convert('L')  
              
            # 转换到1位深度的黑白图像  
            img = img.point(lambda p: 0 if p <= 127 else 255, '1')  
              
            # 保存修改后的图像到原文件  
            img.save(file_path)  
            print(f"Converted {file_path} to 1-bit image")  
  
# 使用函数  
convert_to_1bit(r'predictOutputDir\infers\infers')