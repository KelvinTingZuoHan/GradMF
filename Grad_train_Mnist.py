import sys
sys.path.append('../..')
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from Grad_mf.Vector_Projection_circulate import Grad
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import math
from dataLoader.MnistDataLoader import CustomMNIST,MNiST_DataLoader



 #Mnist单数据集内实验1      
train_img  = MNiST_DataLoader.read_image("./data/MNIST/raw/train-images-idx3-ubyte")
train_label = MNiST_DataLoader.read_label("./data/MNIST/raw/train-labels-idx1-ubyte")
# test_img = read_image("mnist/test/t10k-images.idx3-ubyte")
# test_label = read_label("mnist/test/t10k-labels.idx1-ubyte")
# print(type(train_img))
train_img_A = []
train_label_A = []
train_img_B = []
train_label_B = []

for idx,num in enumerate(train_label) :
    if num>=5 :
        train_label_A.append(num)
        train_img_A.append(train_img[idx])
    else:
        train_label_B.append(num)
        train_img_B.append(train_img[idx])

mnistc19 = CustomMNIST(train_img_A,train_label_A)
mnistc0 = CustomMNIST(train_img_B, train_label_B)

train_size_A = int(len(mnistc19) * 0.8)
test_size_A = len(mnistc19) - train_size_A
train_set_A, test_set_A = random_split(mnistc19, [train_size_A, test_size_A])

train_size_B = int(len(mnistc0) * 0.005)
test_size_B = len(mnistc0) - train_size_B

train_set_B, test_set_B = random_split(mnistc0, [train_size_B, test_size_B])

mnistc19_dl_train = torch.utils.data.DataLoader(train_set_A, batch_size=256, shuffle=False)
mnistc0_dl_train = torch.utils.data.DataLoader(train_set_B, batch_size=32, shuffle=False)

mnistc19_dl_test = torch.utils.data.DataLoader(test_set_A ,batch_size=256, shuffle=False)
mnistc0_dl_test = torch.utils.data.DataLoader(test_set_B, batch_size=32, shuffle=False)










#MNIST和SVHN数据集实验2
# train_dataset = torchvision.datasets.SVHN(
#     root='./dataset/SVHN/SVHN',
#     split='train',
#     download=False,
#     transform=torchvision.transforms.ToTensor()
# )
 
# test_dataset = torchvision.datasets.SVHN(
#     root='./dataset/SVHN/SVHN',
#     split='test',
#     download=False,
#     transform=torchvision.transforms.ToTensor()
# )
 
# # define train loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True,batch_size=256)
 
# # define test loader
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size=256)
 
# images_train, labels_train = next(iter(train_loader))
# images_test, labels_test = next(iter(test_loader))


class MNIST_Net(nn.Module):
 
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            # torch.nn.Conv2d(3, 16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),#2
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            # torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),#2
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size,-1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）



import os
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
epoch = 20

model = MNIST_Net()
model2 = MNIST_Net()

lossfun = torch.nn.CrossEntropyLoss()
lossfun_2 = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
Grad_optimizer = Grad(optimizer2)
torch.cuda.is_available()


#形成A和B的训练集和测试集，且A的训练集较大B的测试集较大，以达到大量A训练来测试B数据集
best_accuracy_A = 0
best_accuracy_B =0 
for ep in range(epoch):
    print(ep)
    running_loss = 0.0
    if ep%3 != 0 or ep==0:
        if os.path.exists("./Model/Model_Mnist/best_model_A.pth")==True:
            model.load_state_dict(torch.load("./Model/Model_Mnist/best_model_A.pth"))
            print("Loading the Best model")
        for batch_idx,(data,target) in enumerate(mnistc19_dl_train):#根据实验内容放置对应的loader
            optimizer.zero_grad()
            model.to(DEVICE)
            model.train()
            

         
            target,data=target.to(DEVICE),data.to(DEVICE)
            output_A = model(data.to(torch.float32))
            # print(output_A.shape,target.shape)
            loss_A = lossfun(output_A,target)
            # print(loss_A)
            loss_A.backward(retain_graph=True)
            optimizer.step()  # Crashes here..
            correct_A = 0
            total_A = 0
            
            with torch.no_grad():
                for data, target in mnistc19_dl_test:#根据实验内容放置对应的loader
                    model.to(DEVICE)
                    model.eval()
                    target,data=target.to(DEVICE),data.to(DEVICE)
                    output = model(data.to(torch.float32))
                    _, predictions_A = torch.max(output.data, 1)
                    
                    total_A += target.size(0)
                    correct_A += (predictions_A == target).sum().item()
                    
            accuracy_A = 100 * correct_A / total_A   
           
        # Save the model if it has the best accuracy so far
        # if (ep+1)%3!= 0 or ep==0:
                
            if accuracy_A > best_accuracy_A :
                torch.save(model.state_dict(), './Model/Model_Mnist/best_model_A.pth')
                best_accuracy_A = accuracy_A
        # else :
        #     if accuracy_A > best_accuracy_A :
        #         torch.save(model.state_dict(), 'best_model_B.pth')
        #         best_accuracy_A = accuracy_A
        print(f'Epoch :{ep}  Test Accuracy_A: {accuracy_A}% Best Accuracy_A:{best_accuracy_A}%')
    else:
        print("Adjusting the model based on B")
        model2.load_state_dict(torch.load("./Model/Model_Mnist/best_model_A.pth"))
        print("Loading the Best model in A")

        for batch_idx,(data,target) in enumerate(mnistc19_dl_train):
            optimizer.zero_grad()
            model.to(DEVICE)
            model.train()
            target,data=target.to(DEVICE),data.to(DEVICE)
            output_A = model(data.to(torch.float32))
            loss_A = lossfun(output_A,target)
            
        for batch_idx,(data,target) in enumerate(mnistc0_dl_train):
            Grad_optimizer = Grad(optimizer2)
            Grad_optimizer.zero_grad()
            
            model2.to(DEVICE)
            model2.train()
            target,data=target.to(DEVICE),data.to(DEVICE)
            output_B = model2(data.to(torch.float32))

            loss_B = lossfun_2(output_B,target)
            
            # print(loss_A)
            # print(type(loss_A))
            #Grad_optimizer.pc_backward([loss_A],[loss_B])
            Grad_optimizer.pc_backward(loss_A,loss_B)
            Grad_optimizer.step()
            correct_B = 0
            total_B = 0

            with torch.no_grad():
                for data, target in mnistc0_dl_test:
                    model2.to(DEVICE)
                    model2.eval()
                    target,data=target.to(DEVICE),data.to(DEVICE)
                    output = model2(data.to(torch.float32))
                    _, predictions_B = torch.max(output.data, 1)
                    total_B += target.size(0)
                    correct_B += (predictions_B == target).sum().item()
            accuracy_B = 100 * correct_B / total_B  
                  
        # Save the model if it has the best accuracy so far
        # print(accuracy_B)
        if accuracy_B > best_accuracy_B:
            torch.save(model2.state_dict(), './Model/Model_Mnist/best_model_A.pth')
            best_accuracy_B = accuracy_B   
        print(f'Epoch :{ep}  Test_B Accuracy: {accuracy_B}% Best Accuracy_B:{best_accuracy_B}%')



print(f'End of training,CurrentBest Accuracy_A:{best_accuracy_A}%    Best Accuracy_B:{best_accuracy_B}%')


