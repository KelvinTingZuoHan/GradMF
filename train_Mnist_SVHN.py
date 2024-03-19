
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
import os
from Net import ResNet
from Net import Basicblock
import cv2
import scipy.io as scio
import struct
from einops import rearrange

# def trans(image, label, save):#image位置，label位置和转换后的数据保存位置
#     if 'train' in os.path.basename(image):
#         prefix = 'train'
#     else:
#         prefix = 'test'
 
#     labelIndex = 0
#     imageIndex = 0
#     i = 0
#     lbdata = open(label, 'rb').read()
#     magic, nums = struct.unpack_from(">II", lbdata, labelIndex)
#     labelIndex += struct.calcsize('>II')
 
#     imgdata = open(image, "rb").read()
#     magic, nums, numRows, numColumns = struct.unpack_from('>IIII', imgdata, imageIndex)
#     imageIndex += struct.calcsize('>IIII')
 
#     for i in range(nums):
#         label = struct.unpack_from('>B', lbdata, labelIndex)[0]
#         labelIndex += struct.calcsize('>B')
#         im = struct.unpack_from('>784B', imgdata, imageIndex)
#         imageIndex += struct.calcsize('>784B')
#         im = np.array(im, dtype='uint8')
#         img = im.reshape(28, 28)
#         save_name = os.path.join(save, '{}_{}_{}.jpg'.format(prefix, i, label))
#         cv2.imwrite(save_name, img)
        
# train_images = './dataset/MNIST/raw/train-images-idx3-ubyte'
# train_labels = './dataset/MNIST/raw/train-labels-idx1-ubyte'
# test_images ='./dataset/MNIST/raw/t10k-images-idx3-ubyte'
# test_labels = './dataset/MNIST/raw/t10k-labels-idx1-ubyte'
# #此处是我们将转化后的数据集保存的位置
# save_train ='./dataset/MNIST/train_images/'
# save_test ='./dataset/MNIST/test_images/'
# trans(test_images, test_labels, save_test)
# trans(train_images, train_labels, save_train)



#以MNIST数据为A,SVHN数据为B   
# MNIST_train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./dataset/', train=True, download=False,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=256, shuffle=True)
# MNIST_test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./dataset/', train=False, download=False,
#                                transform=torchvision.transforms.Compose([
#                                    torchvision.transforms.ToTensor(),
#                                    torchvision.transforms.Normalize(
#                                        (0.1307,), (0.3081,))
#                                ])),
#     batch_size=256, shuffle=True)

# SVHN_test_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(
#     root='./dataset/SVHN/SVHN',
#     split='train',
#     download=False,
#     transform=torchvision.transforms.ToTensor()
# ), shuffle=True,batch_size=256)
# SVHN_train_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(
#     root='./dataset/SVHN/SVHN',
#     split='test',
#     download=False,
#     transform=torchvision.transforms.ToTensor()
# ),shuffle=True,batch_size=256)


#以MNIST数据为B,SVHN数据为
MNIST_train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=256, shuffle=True)
MNIST_test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=256, shuffle=True)

SVHN_train_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(
    root='./data/SVHN/SVHN',
    split='train',
    download=False,
    transform=torchvision.transforms.ToTensor()
), shuffle=True,batch_size=256)
SVHN_test_loader = torch.utils.data.DataLoader(torchvision.datasets.SVHN(
    root='./data/SVHN/SVHN',
    split='test',
    download=False,
    transform=torchvision.transforms.ToTensor()
),shuffle=True,batch_size=256)


print(len(MNIST_train_loader),len(MNIST_test_loader),len(SVHN_train_loader),len(SVHN_test_loader))





import os
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
epoch = 100

model = ResNet(Basicblock, [1, 1, 1, 1], 10)
model2 = ResNet(Basicblock, [1, 1, 1, 1], 10)

lossfun = torch.nn.CrossEntropyLoss()
lossfun_2 = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
Grad_optimizer = Grad(optimizer2)
torch.cuda.is_available()


# #形成A和B的训练集和测试集，且A的训练集较大B的测试集较大，以达到大量A训练来测试B数据集
best_accuracy_A = 0
best_accuracy_B =0 
for ep in range(epoch):
    print(ep)
    running_loss = 0.0
    if ep%3 != 0 or ep==0:
        if os.path.exists("./Model/Model_CelebA/best_model_MNISTtoSVHN.pth")==True:
            model.load_state_dict(torch.load("./Model/Model_CelebA/best_model_MNISTtoSVHN.pth"))
            print("Loading the Best model")
        for batch_idx,(data,target) in enumerate(MNIST_train_loader):#根据实验内容放置对应的loader
            
            data = F.interpolate(data, size=[32,32])
            data1 = torch.cat((data,data,data,),dim = 1)
          
            optimizer.zero_grad()
            model.to(DEVICE)
            model.train()

            # print(data.shape)
            target,data1=target.to(DEVICE),data1.to(DEVICE)
            output_A = model(data1.to(torch.float32))
           
            loss_A = lossfun(output_A,target)
         
            loss_A.backward(retain_graph=True)
            optimizer.step()  # Crashes here..
            correct_A = 0
            total_A = 0
            
        with torch.no_grad():
            for data, target in MNIST_test_loader:#根据实验内容放置对应的loader
                data2 = F.interpolate(data, size=[32,32])
                data2 = torch.cat((data,data,data,),dim = 1)
                model.to(DEVICE)
                model.eval()

                target,data2=target.to(DEVICE),data2.to(DEVICE)
                output = model(data2.to(torch.float32))
                _, predictions_A = torch.max(output.data, 1)
                
                total_A += target.size(0)
                correct_A += (predictions_A == target).sum().item()
                
        accuracy_A = 100 * correct_A / total_A   
           
        # Save the model if it has the best accuracy so far
        # if (ep+1)%3!= 0 or ep==0:     
        if accuracy_A > best_accuracy_A :
            torch.save(model.state_dict(), './Model/Model_CelebA/best_model_MNISTtoSVHN.pth')
            best_accuracy_A = accuracy_A
    # else :
        #     torch.save(model.state_dict(), 'best_model_B_3.pth')
        #     if accuracy_A > best_accuracy_A :
        #         torch.save(model.state_dict(), 'best_model_B_3.pth')#存在保存时的逻辑bug，还需优化保存模型策略以及更新准确率逻辑
        #         best_accuracy_A = accuracy_A
                
        print(f'Epoch :{ep}  Test Accuracy_A: {accuracy_A}% Best Accuracy_A:{best_accuracy_A}%')
    else:
        print("Adjusting the model based on B")
        model2.load_state_dict(torch.load("./Model/Model_CelebA/best_model_MNISTtoSVHN.pth"))
        print("Loading the Best model in A")

        for batch_idx,(data,target) in enumerate(MNIST_train_loader):#根据实验内容放置对应的loader
            
            data = F.interpolate(data, size=[32,32])
            data1 = torch.cat((data,data,data,),dim = 1)
            optimizer.zero_grad()
            model.to(DEVICE)
            model.train()
            target,data1=target.to(DEVICE),data1.to(DEVICE)
            output_A = model(data1.to(torch.float32))
            
            loss_A = lossfun(output_A,target)
            
        for batch_idx,(data,target) in enumerate(SVHN_train_loader):
            Grad_optimizer = Grad(optimizer2)
            Grad_optimizer.zero_grad()

            # data = F.interpolate(data, size=[32,32])
            # data1 = torch.cat((data,data,data,),dim = 1)

            model2.to(DEVICE)
            model2.train()
            target,data=target.to(DEVICE),data.to(DEVICE)
            output_B = model2(data.to(torch.float32))
            loss_B = lossfun_2(output_B,target)
            
            # print(loss_A)
            # print(type(loss_A))
            # Grad_optimizer.pc_backward([loss_A],[loss_B],ep)
            Grad_optimizer.pc_backward(loss_A,loss_B)
            Grad_optimizer.step()
            correct_B = 0
            total_B = 0

        with torch.no_grad():
            for data, target in SVHN_test_loader:
                model2.to(DEVICE)
                model2.eval()
                data = F.interpolate(data, size=[32,32])
                data2 = torch.cat((data,data,data,),dim = 1)

                target,data=target.to(DEVICE),data.to(DEVICE)
                output = model2(data.to(torch.float32))
                _, predictions_B = torch.max(output.data, 1)
                total_B += target.size(0)
                correct_B += (predictions_B == target).sum().item()
        accuracy_B = 100 * correct_B / total_B  
                  
        # Save the model if it has the best accuracy so far
        # print(accuracy_B)
        if accuracy_B > best_accuracy_B:
            torch.save(model2.state_dict(), 'best_model_A_3.pth')
            best_accuracy_B = accuracy_B   
        print(f'Epoch :{ep}  Test_B Accuracy: {accuracy_B}% Best Accuracy_B:{best_accuracy_B}%')



print(f'End of training,CurrentBest Accuracy_A:{best_accuracy_A}%    Best Accuracy_B:{best_accuracy_B}%')


