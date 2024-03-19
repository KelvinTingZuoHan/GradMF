import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import transforms
import torch.utils.data as data

import matplotlib.pyplot as plt
from torch.utils.data import random_split
import math
  



class CustomMNIST(data.Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

    
   
class MNiST_DataLoader():
    def read_image(file_path):
            
        with open(file_path,'rb') as f:
            file = f.read()
            img_num = int.from_bytes(file[4:8],byteorder='big') #图片数量
            img_h = int.from_bytes(file[8:12],byteorder='big') #图片h
            img_w = int.from_bytes(file[12:16],byteorder='big') #图片w
            img_data = []
            file = file[16:]
            data_len = img_h*img_w

            for i in range(img_num):
                data = [item/255 for item in file[i*data_len:(i+1)*data_len]]
                img_data.append(np.array(data).reshape(1,img_h,img_w))

        return img_data 
    def read_label(file_path):

        with open(file_path,'rb') as f:
            file = f.read()
            label_num = int.from_bytes(file[4:8],byteorder='big') #label的数量
            file = file[8:]
            label_data = []
            for i in range(label_num):
                label_data.append(file[i])
        return label_data