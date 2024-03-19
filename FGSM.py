

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
pretrained_model = "./Model/Model_CelebA/best_model_MNISTtoSVHN.pth"

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=256, shuffle=True)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = ResNet(Basicblock, [1, 1, 1, 1], 10)
model.load_state_dict(torch.load(pretrained_model))
model.eval()

def fgsm_attack(image, epsilon, data_grad):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image + epsilon*sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image

def test( model, device, test_loader, epsilon ):
 
    # 准确度计数器
    correct = 0
    # 对抗样本
    adv_examples = []
 
    # 循环所有测试集
    for data, target in test_loader:
        # 将数据和标签发送到设备
        data = F.interpolate(data, size=[32,32])
        data1 = torch.cat((data,data,data,),dim = 1)

        data1, target = data1.to(device), target.to(device)
 
        # 设置张量的requires_grad属性。重要的攻击
        data.requires_grad = True
 
        # 通过模型向前传递数据
        output = model(data1)
        init_pred = output.max(1, keepdim=True)[1] # 得到最大对数概率的索引
 
        # 如果最初的预测是错误的，不要再攻击了，继续下一个目标的对抗训练
        if init_pred.item() != target.item():
            continue
 
        # 计算损失
        loss = F.nll_loss(output, target)
 
        # 使所有现有的梯度归零
        model.zero_grad()
 
        # 计算模型的后向梯度
        loss.backward()
 
        # 收集datagrad
        data_grad = data.grad.data
 
        # 调用FGSM攻击
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
 
        # 对受扰动的图像进行重新分类
        output = model(perturbed_data)
 
        # 检查是否成功
        final_pred = output.max(1, keepdim=True)[1] # 得到最大对数概率的索引
        if final_pred.item() == target.item():
            correct += 1
            # 这里都是为后面的可视化做准备
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # 这里都是为后面的可视化做准备
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
 
    # 计算最终精度
    final_acc = correct/float(len(test_loader))
    print("扰动量: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
 
    # 返回准确性和对抗性示例
    return final_acc, adv_examples


accuracies = []
examples = []

epsilons = [0, .05, .1, .15, .2, .25, .3,.35,.4]
# 对每个干扰程度进行测试
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc*100)
    examples.append(ex)
 
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 110, step=10))
plt.xticks(np.arange(0, .5, step=0.05))
def to_percent(temp, position):
    return '%1.0f'%(temp) + '%'
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.title("准确率 vs 扰动量")
plt.xlabel("扰动量")
plt.ylabel("准确率")
plt.show()
