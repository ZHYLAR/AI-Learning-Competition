import copy
import cv2
import os
import shutil
import numpy as np


path = r"train\mask_bin"
files = os.listdir(path)
for file in files:
    name = file.split('.')[0]
    file_path = os.path.join(path,name+'.png')
    img = cv2.imread(file_path)
    # img = cv2.imread(path)
    H,W=img.shape[0:2]
    #print(H,W)

    #img1 = cv2.imread("F:/Deep_Learning/Model/YOLOv8_Seg/Dataset/images/20160222_080933_361_1.jpg")

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,bin_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnt,hit = cv2.findContours(bin_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_KCOS)

    #cv2.drawContours(img1,cnt,-1,(0,255,0),5)

    cnt = list(cnt)
    f = open("train/labels/{}.txt".format(file.split(".")[0]), "a+")
    for j in cnt:
        result = []
        pre = j[0]
        for i in j:
            if abs(i[0][0] - pre[0][0]) > 1 or abs(i[0][1] - pre[0][1]) > 1:# 在这里可以调整间隔点，我设置为1
                pre = i
                temp = list(i[0])
                temp[0] /= W
                temp[1] /= H
                result.append(temp)

                #cv2.circle(img1,i[0],1,(0,0,255),2)

        #print(result)
        #print(len(result))

        # if len(result) != 0:

        if len(result) != 0:
            f.write("0 ")
            for line in result:
                line = str(line)[1:-2].replace(",","")
                # print(line)
                f.write(line+" ")
            f.write("\n")
    f.close()

# import os
# import numpy as np
# from itertools import groupby
# from skimage import morphology, measure
# from PIL import Image
# from scipy import misc
# import cv2
 
# # 因为一张图片里只有一种类别的目标，所以label图标记只有黑白两色
# rgbmask = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
 
 
# # 从label图得到 boundingbox 和图上连通域数量 object_num
# def getboundingbox(image):
#     # mask.shape = [image.shape[0], image.shape[1], classnum]
#     mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
#     mask[np.where(np.all(image == rgbmask[1], axis=-1))[:2]] = 1
#     # 删掉小于10像素的目标
#     mask_without_small = morphology.remove_small_objects(mask, min_size=10, connectivity=2)
#     # 连通域标记
#     label_image = measure.label(mask_without_small)
#     # 统计object个数
#     object_num = len(measure.regionprops(label_image))
#     boundingbox = list()
#     bboxs=list()
#     for region in measure.regionprops(label_image):  # 循环得到每一个连通域bbox
#         b=[region.bbox[1],region.bbox[3],region.bbox[0],region.bbox[2]]
#         bbox=convert([4096,4096],b)
#         bboxs.append(bbox)
#         boundingbox.append(region.bbox)#
#     return object_num, boundingbox,bboxs
 
# def convert(size, box):
#     dw = 1. / size[0]
#     dh = 1. / size[1]
#     x = (box[0] + box[1]) / 2.0
#     y = (box[2] + box[3]) / 2.0
#     w = box[1] - box[0]
#     h = box[3] - box[2]
#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return (x, y, w, h)
 
# def mask_to_yolo(mask_dir,save_dir):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     imgs=os.listdir(mask_dir)
#     for img in imgs:
#         img_path=os.path.join(mask_dir,img)
#         img0 = cv2.imread(img_path)
#         out_file = open(save_dir+'/'+img.split('.')[0]+'.txt', 'w')
#         object_num, boundingbox, bboxs = getboundingbox(img0)
#         for i in bboxs:
#             out_file.write('0' + " " + " ".join([str(a) for a in i]) + '\n')
 
 
 
# if __name__ == '__main__':
#     mask_to_yolo(r'train\mask_bin','train\labels')
#     # imgdir='train\mask_bin\A-1.png'
#     # img=cv2.imread(imgdir)
#     # out_file = open('1.txt', 'w')
#     # object_num, boundingbox,bboxs=getboundingbox(img)
    
#     # print(object_num,boundingbox)
#     # for i in bboxs:
#     #     out_file.write('0' + " " + " ".join([str(a) for a in i]) + '\n')
 
 