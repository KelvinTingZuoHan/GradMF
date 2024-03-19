import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import math
from PIL import Image
import os

class CelebADataset(data.Dataset): # 继承Dataset类
    def __init__(self, img_path,label_path,transform = transforms.Compose([
                                    transforms.CenterCrop(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                                                        ])): # 定义txt_path参数
        lp = open(label_path, 'r') # 读取txt文件
        
        labels = lp.read()
        labels= eval(labels)
        # print(len(labels))
        # print(labels)
        labels_new = np.delete(labels,0,axis=1)
        # print(type(labels),len(labels))
    
        labels_2 = []
        for i in labels_new:
            i = np.array(i).astype(dtype=int).tolist()
            labels_2.append(i)
            # print(len(labels_2),type(labels_2))
        # for num in labels_new :
        #     # print (num)
        #     # print(labels_new[n])
        #     labels_1 = [float(nums) for nums in labels_new[n]]
        #     print(labels_1)
        #     labels_2 = np.append(labels_2,labels_1)
        #     n = n+1
        #     print(labels_2)    

        # print(labels_2[0,1])
        
        # print(labels)
        
    


        file_list = os.listdir(img_path)
        
        
        file_list.sort(key=lambda x:int(x.split('.')[0]))
        # print(type(file_list))
        file_path_list = [os.path.join(img_path, img) for img in file_list]
     
        imgs = file_path_list  # 定义imgs的列表
        # print(imgs)

       

        # print (labels)
        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.labels = labels_2
        # print(len(self.imgs),len(self.labels))
        # self.target_transform = target_transform
        
    def __getitem__(self, index):
        # print(self.imgs)
        imag = self.imgs[index]
        
        img = Image.open(imag).convert('RGB')
        # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1   
       
        label = self.labels[index]
        # print(img,label)

        img = self.transform(img)   # 在这里做transform，转为tensor等等
        # label = self.transform(label)
        
        
        
        label_1=torch.tensor(label)
        label = label_1.ne(-1).long() * label_1
        
        # print(img,label,label_1)
        # print(type(img),type(label))
        
        return img, label

    def __len__(self):
        return len(self.imgs)   # 返回图片的长度





# with open('./data/celeba/list_attr_celeba.txt', "r") as Attr_file:
#     Attr_info = Attr_file.readlines()
#     Attr_info = Attr_info[2:]
#     index = 0
# x = CelebaDataset()
# print(x)