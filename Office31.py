import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import transforms , datasets
import torch.utils.data as data
from Grad_mf.Vector_Projection_circulate import Grad as Grad_C
from Grad_mf.Vector_Projection import Grad as Grad_N

import matplotlib.pyplot as plt
from torch.utils.data import random_split
import math
import os
import torch.optim as optim
import cv2
import scipy.io as scio
import struct
from einops import rearrange
from imageio.v2 import imread, imsave


def load_data(root_path, batch_size ,phase):
    data = datasets.ImageFolder(root=os.path.join(root_path), 
                                transform= transforms.Compose(
                                    [transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
                                                        ]))
    if phase == 'Src':
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        data_train, data_test = torch.utils.data.random_split(data, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    else:
        train_size = int(0.99 * len(data))
        test_size = len(data) - train_size
        data_train, data_test = torch.utils.data.random_split(data, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
   
    return train_loader, test_loader

def load_model(name='alexnet'):
    if name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        n_features = model.classifier[6].in_features
        fc = torch.nn.Linear(n_features, 31)
        model.classifier[6] = fc
    elif name == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
        n_features = model.fc.in_features
        fc = torch.nn.Linear(n_features, 31)#31类，最后全连接层
        model.fc = fc
    model.fc.weight.data.normal_(0, 0.005)
    model.fc.bias.data.fill_(0.1)
    return model



train_A,test_A = load_data('/root/Test/czh/GradMF/data/office31/amazon',32,phase="Src")
train_B,test_B =load_data('/root/Test/czh/GradMF/data/office31/webcam',32,phase='Tar')
print(len(train_A),len(test_A))
print(len(train_B),len(test_B))


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

model = load_model('resnet').to(DEVICE)
model2 = load_model('resnet').to(DEVICE)

Epoch = 50

loss_criterion = nn.CrossEntropyLoss() #定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum = 0.9 )#定义优化器
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

    correct_train = 0 #模型分类对的训练图片
    running_loss = 0 #训练集上的loss
    running_test_loss = 0 #测试集上的loss
    total_test = 0 #测试的图片总数
    correct_test = 0 #分类对的测试图片数
    model.train() #训练模式
    if epoch%2 != 0 or epoch==0:
        
        if os.path.exists("./Model/Model_Office31/best_model_AtoW.pth")==True:
            model.load_state_dict(torch.load("./Model/Model_Office31/best_model_AtoW.pth"))
            print("Loading the Best model")

        model.train()
        for data, target in train_A:
            data = data.to(device=DEVICE)
            target = target.to(device=DEVICE)
            
            # print(target.shape,data.shape)
            # print(data.shape,data)
            score = model(data)
            loss_A = loss_criterion(score, target)
           
            # print(score,target,loss)


            running_loss += loss_A.item()
            
            optimizer.zero_grad()
            
            loss_A.backward()
            
            optimizer.step()
            
            
            # print(score,sigmoid_logits)
            preds = torch.max(score, 1)[1] #使结果变为true,false的数组
          
        model.eval()
        correct = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_B:
                data, target = data.to(DEVICE), target.to(DEVICE)
                s_output = model(data)
                loss = criterion(s_output, target)
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target)
        test_acc = correct.double() / len(test_A.dataset)

        if test_acc > best_accuracy_A:
            best_accuracy_A = test_acc
            torch.save(model.state_dict(), './Model/Model_Office31/best_model_AtoW.pth')

        print(f"For epoch : {epoch} test loss: {running_loss/len(test_A.dataset)}")
        print(f'Epoch :{epoch}  Test Accuracy_A: {test_acc*100}% Best Accuracy_A:{best_accuracy_A*100}%')
    # else:
    #     print("Adjusting the model based on B")
    #     model2.load_state_dict(torch.load("./Model/Model_Office31/best_model_AtoW.pth"))
    #     print("Loading the Best model in A")
    #     model2.train()
    #     for data, target in train_A:
            
    #         data = data.to(device=DEVICE)
    #         target = target.to(device=DEVICE)
            
    #         # print(target_2.shape)
    #         score = model2(data)
    #         # print(score_2.shape)
    #         loss_A = loss_criterion(score, target)

    #     for data, target in train_B:
    #         data = data.to(device=DEVICE)
    #         target = target.to(device=DEVICE)
            
    #         # print(data.shape,data)
    #         score = model2(data)
    #         loss_B = loss_criterion(score, target)
    #         # print(score,target,loss)
    #         running_loss += loss_B.item()
    #         Grad_optimizer.zero_grad()
    #         Grad_optimizer.pc_backward(loss_A,loss_B,epoch)
    #         Grad_optimizer.step()

    #     model2.eval()#测试模式
    #     correct = 0
    #     criterion = torch.nn.CrossEntropyLoss()
    #     with torch.no_grad():
    #         for data, target in test_B:
    #             data, target = data.to(DEVICE), target.to(DEVICE)
    #             s_output = model2(data)
    #             loss = criterion(s_output, target)
    #             pred = torch.max(s_output, 1)[1]
    #             correct += torch.sum(pred == target)
    #         test_acc_B = correct.double() / len(test_B.dataset)

    #         if test_acc_B > best_accuracy_B:
    #             best_accuracy_B = test_acc_B
    #             torch.save(model.state_dict(), './Model/Model_Office31/best_model_AtoW.pth')

    #         print(f"For epoch : {epoch} test loss: {running_loss/len(test_B.dataset)}")
    #         print(f'Epoch :{epoch}  Test Accuracy_B: {test_acc_B*100}% Best Accuracy_B:{best_accuracy_B*100}%')

print(f'End of training,CurrentBest Accuracy_A:{best_accuracy_A}%    Best Accuracy_B:{best_accuracy_B}%')