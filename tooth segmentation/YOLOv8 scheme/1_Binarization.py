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
convert_to_1bit(r'E:\desk\tooth-2d\predictOutputDir\infers\infers')