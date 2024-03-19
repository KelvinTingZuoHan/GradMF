import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from Grad_mf.Vector_Projection_circulate import Grad as Grad_C
from Grad_mf.Vector_Projection import Grad as Grad_N
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import math
import os
from Net import ResNet
from Net import SlimNet
from Net import Basicblock
from Net import Mnist_Net
import cv2
import scipy.io as scio
import struct
from einops import rearrange
from imageio.v2 import imread, imsave
from dataLoader.CelebADataloader import CelebADataset


# data_dir = './data/celeba'        # 将整理好的数据放在这个文件夹下
# male_dir = './data/celeba/DataSet_A'
# female_dir = './data/celeba/DataSet_B'
# if not os.path.exists(data_dir):
#     os.mkdir(data_dir)
# if not os.path.exists(male_dir):
#     os.mkdir(male_dir)
# if not os.path.exists(female_dir):
#     os.mkdir(female_dir)



# Attr_type_1 = 1 # 5_o_Clock_Shadow 胡子，（清晨刮脸的人傍晚已长出的短髭 ）
# Attr_type_2 = 2 # Arched_Eyebrows 柳叶眉 
# Attr_type_3 = 3 #Attractive 有魅力的 
# Attr_type_4 = 4 #Bags_Under_Eyes 眼袋 
# Attr_type_5 = 5 #Bald 秃头的 
# Attr_type_6 = 6 #Bangs 刘海 
# Attr_type_7 = 7 #Big_Lips 大嘴唇 
# Attr_type_8 = 8 #Big_Nose 大鼻子 
# Attr_type_9 = 9 #Black_Hair 黑发 
# Attr_type_10= 10 #Blond_Hair 金发 
# Attr_type_11= 11 #Blurry 睡眼惺松的 
# Attr_type_12= 12 # Brown_Hair 棕发 
# Attr_type_13= 13 #Bushy_Eyebrows 浓眉 
# Attr_type_14= 14 #Chubby 丰满的 
# Attr_type_15= 15 #Double_Chin 双下巴 
# Attr_type_16= 16 #Eyeglasses 眼镜 
# Attr_type_17= 17 #Goatee 山羊胡子 
# Attr_type_18= 18 #Gray_Hair 白发，灰发 
# Attr_type_19= 19 #Heavy_Makeup 浓妆 
# Attr_type_20= 20 #High_Cheekbones 高颧骨 
# Attr_type_21= 21 #Male 男性 
# Attr_type_22= 22 #Mouth_Slightly_Open 嘴轻微的张开
# Attr_type_23= 23 #Mustache 胡子 
# Attr_type_24= 24 #Narrow_Eyes 窄眼 
# Attr_type_25= 25 #No_Beard 没有胡子
# Attr_type_26= 26 #Oval_Face 瓜子脸，鹅蛋脸 
# Attr_type_27= 27 #Pale_Skin 白皮肤 
# Attr_type_28= 28 #Pointy_Nose 尖鼻子
# Attr_type_29= 29 #Receding_Hairline 发际线; 向后梳得发际线 
# Attr_type_30= 30 #Rosy_Cheeks 玫瑰色的脸颊 
# Attr_type_31= 31 #Sideburns 连鬓胡子，鬓脚 
# Attr_type_32= 32 #Smiling 微笑的 
# Attr_type_33= 33 #Straight_Hair 直发 
# Attr_type_34= 34 #Wavy_Hair 卷发; 波浪发 
# Attr_type_35= 35 #Wearing_Earrings 戴耳环 
# Attr_type_36= 36 #Wearing_Hat 带帽子 
# Attr_type_37= 37 #Wearing_Lipstick 涂口红 
# Attr_type_38= 38 #Wearing_Necklace 带项链 
# Attr_type_39= 39 #Wearing_Necktie 戴领带 
# Attr_type_40= 40 #Young 年轻人 


# WIDTH = 128
# HEIGHT = 128  

# def read_process_save(read_path, save_path):
#     image = imread(read_path)
#     h = image.shape[0]
#     w = image.shape[1]
#     if h > w:
#         image = image[h // 2 - w // 2: h // 2 + w // 2, :, :]
#     else:
#         image = image[:, w // 2 - h // 2: w // 2 + h // 2, :]    
#     image = cv2.resize(image, (WIDTH, HEIGHT))
#     imsave(save_path, image)


# with open('./data/celeba/list_attr_celeba.txt', "r") as Attr_file:
#     Attr_info = Attr_file.readlines()
#     Attr_info = Attr_info[2:]
# # print(Attr_info[:1])

# target = 'Male'
# label_A = []
# label_B = []
# label_A_1 = []
# label_B_1 = []
# with open('./data/celeba/list_attr_celeba.txt', "r") as Attr_file:
#     Attr_info = Attr_file.readlines()
#     Attr_info = Attr_info[2:]
#     index = 0
#     print("进入区分程序")
#     for line in Attr_info:
#         index += 1
#         info = line.split()
#         filename = info[0]
#         filepath_old = os.path.join("./data/celeba/img_align_celeba", filename)
#         if os.path.isfile(filepath_old):
            
#             if int(info[Attr_type_21]) == 1 and int(info[Attr_type_39]) == -1 and int(info[Attr_type_24]) == 1:
#                 read_process_save(os.path.join("./data/celeba/img_align_celeba", filename), os.path.join(male_dir, filename)) # 男

#                 label_A_1.append(info[0])
#                 label_A_1.append(info[Attr_type_21])
#                 label_A_1.append(info[Attr_type_39])
#                 label_A_1.append(info[Attr_type_24])
#                 label_A.append(label_A_1)
#                 label_A_1 = []
#                 # label_A.append(info)
#                 # print(label_A)
#             elif int(info[Attr_type_21])== -1 and int(info[Attr_type_3])== -1 and int(info[Attr_type_4]) == -1:
#                 read_process_save(os.path.join("./data/celeba/img_align_celeba", filename), os.path.join(female_dir, filename)) # 女
#                 label_B_1.append(info[0])
#                 label_B_1.append(info[Attr_type_21])
#                 label_B_1.append(info[Attr_type_3])
#                 label_B_1.append(info[Attr_type_4])
#                 label_B.append(label_B_1)
#                 label_B_1 = []
#                 # label_B.append(info)
        

#     num_1 = len(os.listdir(male_dir))
#     num_2 = len(os.listdir(female_dir))

#     with open('./data/celeba/DataSet_A_label.txt','w') as f:
#         f.write(str(label_A))
#     print(len(label_A))
#     with open('./data/celeba/DataSet_B_label.txt','w') as f:
#         f.write(str(label_B))
#     print(len(label_B))
#     print(f"区分完毕,A数据集:{num_1},B数据集:{num_2}" )




# train_dataset = torchvision.datasets.CelebA(root='./data/',
#                                 split='train',
#                                 transform = transforms.Compose([
#                                     transforms.CenterCrop(128),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                                         std=[0.5, 0.5, 0.5])]),
#                                 download=False)
# test_dataset = torchvision.datasets.CelebA(root="./data/",
#                                 split='test',
#                                 transform = transforms.Compose([
#                                     transforms.CenterCrop(128),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                                         std=[0.5, 0.5, 0.5])]),
#                                 download=False)
train_A =CelebADataset('/root/test/szh/zuohan/GradMF/data/celeba/DataSet_B',"/root/test/szh/zuohan/GradMF/data/celeba/DataSet_B_label.txt")
test_A = CelebADataset('/root/test/szh/zuohan/GradMF/data/celeba/DataSet_B',"/root/test/szh/zuohan/GradMF/data/celeba/DataSet_B_label.txt")

train_B =CelebADataset('/root/test/szh/zuohan/GradMF/data/celeba/DataSet_A',"/root/test/szh/zuohan/GradMF/data/celeba/DataSet_A_label.txt")
test_B = CelebADataset('/root/test/szh/zuohan/GradMF/data/celeba/DataSet_A',"/root/test/szh/zuohan/GradMF/data/celeba/DataSet_A_label.txt")

train_size_A = int(len(train_A) * 0.8)
test_size_A = len(train_A) - train_size_A
train_set_A, test_set_A = random_split(train_A, [train_size_A, test_size_A])
print(len(train_set_A),len(test_set_A))

train_size_B = int(len(train_B) * 0.01)
test_size_B = len(train_B) - train_size_B
train_set_B, test_set_B = random_split(train_B, [train_size_B, test_size_B])
print(len(train_set_B),len(test_set_B))
 
train_A_dataloader = torch.utils.data.DataLoader(dataset=train_set_A, shuffle=True, batch_size=128,num_workers=4)
test_A_dataloader = torch.utils.data.DataLoader(dataset=test_set_A, shuffle=False, batch_size=128,num_workers=4)

train_B_dataloader = torch.utils.data.DataLoader(dataset=train_set_B, shuffle=True, batch_size=128,num_workers=4)
test_B_dataloader = torch.utils.data.DataLoader(dataset=test_set_B, shuffle=False, batch_size=128,num_workers=4)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# model = ResNet(Basicblock, [1, 1, 1, 1], 40)
model = SlimNet(num_classes=3)
model2 = SlimNet(num_classes=3)
# model = Mnist_Net()
model = model.to(DEVICE)
model2 = model2.to(DEVICE)
Epoch = 50

loss_criterion = nn.BCEWithLogitsLoss() #定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) #定义优化器
Grad_optimizer = Grad_C(optimizer)
# best_acc = 0.90325 #最好的在测试集上的准确度，可手动修改


#形成A和B的训练集和测试集，且A的训练集较大B的测试集较大，以达到大量A训练来测试B数据集
best_accuracy_A = 0
best_accuracy_B =0 
seed = 18203861252700 #固定起始种子
for epoch in range(Epoch): #训练五十轮
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True#以上这些都是企图固定种子，但经过测试只能固定起始种子，可删掉
    total_train = 0 #总共的训练图片数量，用来计算准确率
    correct_train = 0 #模型分类对的训练图片
    running_loss = 0 #训练集上的loss
    running_test_loss = 0 #测试集上的loss
    total_test = 0 #测试的图片总数
    correct_test = 0 #分类对的测试图片数
    model.train() #训练模式
    if epoch%2 != 0 or epoch==0:
        if os.path.exists("./Model/Model_CelebA/best_model_A_C_Label3.pth")==True:
            model.load_state_dict(torch.load("./Model/Model_CelebA/best_model_A_C_Label3.pth"))
            print("Loading the Best model")

        for data, target in train_A_dataloader:
            data = data.to(device=DEVICE)
            target = target.type(torch.DoubleTensor).to(device=DEVICE)
            
            # print(data.shape,data)
            score = model(data)
            loss_A = loss_criterion(score, target)
           
            # print(score,target,loss)


            running_loss += loss_A.item()
            
            optimizer.zero_grad()
            
            loss_A.backward()
            
            optimizer.step()
            
            sigmoid_logits = torch.sigmoid(score)
            # print(score,sigmoid_logits)
            predictions = sigmoid_logits > 0.5 #使结果变为true,false的数组
            total_train += target.size(0) * target.size(1)
            correct_train += (target.type(predictions.type()) == predictions).sum().item()

        model.eval() #测试模式
        with torch.no_grad():
            for batch_idx, (images,labels) in enumerate(test_A_dataloader):
                images, labels = images.to(DEVICE), labels.type(torch.DoubleTensor).to(DEVICE)
                logits = model.forward(images)

                test_loss = loss_criterion(logits, labels)
                running_test_loss += test_loss.item()

                sigmoid_logits = torch.sigmoid(logits)

                predictions = sigmoid_logits > 0.5

                total_test += labels.size(0) * labels.size(1)

                correct_test += (labels.int() == predictions.int()).sum().item()

        test_acc = correct_test/total_test

        if test_acc > best_accuracy_A:
            best_accuracy_A = test_acc
            torch.save(model.state_dict(), './Model/Model_CelebA/best_model_A_C_Label3.pth')
        # print(f"For epoch : {epoch} training loss: {running_loss/len(train_A_dataloader)}")
        # print(f'train accruacy is {correct_train*100/total_train}%')
        print(f"For epoch : {epoch} test loss: {running_test_loss/len(test_A_dataloader)}")
        print(f'Epoch :{epoch}  Test Accuracy_A: {test_acc*100}% Best Accuracy_A:{best_accuracy_A*100}%')
    else:
        print("Adjusting the model based on B")
        model2.load_state_dict(torch.load("./Model/Model_CelebA/best_model_A_C_Label3.pth"))
        print("Loading the Best model in A")

        for data, target in train_A_dataloader:
            
            data = data.to(device=DEVICE)
            target = target.type(torch.DoubleTensor).to(device=DEVICE)
            
            # print(target_2.shape)
            score = model2(data)
            # print(score_2.shape)
            loss_A = loss_criterion(score, target)

        for data, target in train_B_dataloader:
            data = data.to(device=DEVICE)
            target = target.type(torch.DoubleTensor).to(device=DEVICE)
            
            # print(data.shape,data)
            score = model2(data)
            loss_B = loss_criterion(score, target)
            # print(score,target,loss)
            running_loss += loss_B.item()
            optimizer.zero_grad()
            Grad_optimizer.pc_backward(loss_A,loss_B,epoch)
            optimizer.step()
            sigmoid_logits = torch.sigmoid(score)
            # print(score,sigmoid_logits)
            predictions = sigmoid_logits > 0.5 #使结果变为true,false的数组
            total_train += target.size(0) * target.size(1)
            correct_train += (target.type(predictions.type()) == predictions).sum().item()

        model.eval() #测试模式
        with torch.no_grad():
            for batch_idx, (images,labels) in enumerate(test_B_dataloader):
                images, labels = images.to(DEVICE), labels.type(torch.DoubleTensor).to(DEVICE)
                logits = model2.forward(images)

                test_loss = loss_criterion(logits, labels)
                running_test_loss += test_loss.item()

                sigmoid_logits = torch.sigmoid(logits)

                predictions = sigmoid_logits > 0.5

                total_test += labels.size(0) * labels.size(1)

                correct_test += (labels.int() == predictions.int()).sum().item()

        test_acc_B = correct_test/total_test

        if test_acc_B > best_accuracy_B:
            best_accuracy_B = test_acc_B
            torch.save(model.state_dict(), './Model/Model_CelebA/best_model_A_C_Label3.pth')
         # print(f"For epoch : {epoch} training loss: {running_loss/len(train_A_dataloader)}")
        # print(f'train accruacy is {correct_train*100/total_train}%')
        print(f"For epoch : {epoch} test loss: {running_test_loss/len(test_A_dataloader)}")
        print(f'Epoch :{epoch}  Test Accuracy_B: {test_acc_B*100}% Best Accuracy_B:{best_accuracy_B*100}%')

print(f'End of training,CurrentBest Accuracy_A:{best_accuracy_A}%    Best Accuracy_B:{best_accuracy_B}%')
