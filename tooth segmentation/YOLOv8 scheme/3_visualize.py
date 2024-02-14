# from PIL import Image,ImageDraw 
# import os
# from pathlib import Path 
# from shutil import copyfile 
# from tqdm import tqdm
# import numpy as np 

def get_labels_polys(img_path,gt_path):
    img = Image.open(img_path)
    w,h = img.size  
    with open(gt_path, 'r') as fl:
        lines = [x.rstrip() for x in fl.readlines()]
    str_data = [x.split(' ') for x in lines]
    relative_polys = [[float(x) for x in arr[1:]] for arr in str_data]
    labels = [int(arr[0]) for arr in str_data]
    polys = [ [x*w if i%2==0 else x*h  for i,x in enumerate(arr)]  for arr in relative_polys]
    return labels,polys

def plot_polys(image, polys):
    image_result = image.copy()
    draw = ImageDraw.Draw(image_result) 
    for poly in polys:
        if len(poly) >= 6:  # 假设每个顶点包含两个坐标（x, y），所以至少需要6个数值来构成一个多边形
            draw.polygon(poly, fill="cyan", outline="red") 
        else:
            print(f"Skipping polygon because it contains less than two coordinates: {poly}")
    return image_result 


# from pathlib import Path

# root_path = 'yolo_dataset'

# data_root = Path(root_path)
# val_imgs = [str(x) for x in (data_root/'images'/'val').rglob("*.png") if 'checkpoint' not in str(x)]

# img_path = val_imgs[2] 
# gt_path = img_path.replace('images','labels').replace('.png','.txt')


# labels,polys = get_labels_polys(img_path,gt_path)
# # plot_polys(Image.open(img_path),polys)
# image_result = plot_polys(Image.open(img_path),polys)
# image_result.show()

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 确保已有的函数定义不变

def visualize_polys_in_grid(img_paths, gt_paths, ncols=3):
    fig, axes = plt.subplots(nrows=int(np.ceil(len(img_paths) / ncols)), ncols=ncols, figsize=(15, 15))

    for idx, (img_path, gt_path) in enumerate(zip(img_paths, gt_paths)):
        row_idx = idx // ncols
        col_idx = idx % ncols
        
        # 获取图片和多边形数据
        labels, polys = get_labels_polys(img_path, gt_path)
        image = Image.open(img_path)

        # 绘制并显示带有多边形的图片
        image_with_polys = plot_polys(image, polys)
        
        # 将PIL图片转换成matplotlib可显示格式
        axes[row_idx, col_idx].imshow(np.array(image_with_polys))
        axes[row_idx, col_idx].axis('off')  # 不显示坐标轴

    plt.tight_layout()
    plt.show()

# 获取验证集图片和对应的标签文件路径
data_root = Path('yolo_dataset')
val_imgs = [str(x) for x in (data_root/'images'/'val').rglob("*.png") if 'checkpoint' not in str(x)]
gt_paths = [img_path.replace('images', 'labels').replace('.png', '.txt') for img_path in val_imgs]

# 选择部分或全部图片进行可视化
# 这里假设只取前n个图片作为示例
n_samples = min(9, len(val_imgs))  # 可视化一个3x3的网格
sample_val_imgs = val_imgs[:n_samples]
sample_gt_paths = gt_paths[:n_samples]

visualize_polys_in_grid(sample_val_imgs, sample_gt_paths)