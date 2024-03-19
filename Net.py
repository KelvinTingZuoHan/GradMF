import sys
import torch
import torch.nn as nn
import torch.nn.functional as F




class Mnist_Net(nn.Module):
 
    def __init__(self):
        super(Mnist_Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),#2
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),#2
        )
        self.fc = torch.nn.Sequential(
            # torch.nn.Linear(800, 120),
            # torch.nn.Linear(120, 10),
            torch.nn.Linear(26912, 40)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size,-1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        # print(x.shape)
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
    

class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
 
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
 
        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)
 
        self.outlayer = nn.Linear(64, num_classes)#64,1024
 
    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)                       # [200, 64, 28, 28]
        x = self.block2(x)                       # [200, 128, 14, 14]
        x = self.block3(x)                       # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)                   # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)               # [200,256]
        # print(x.shape)
        
        out = self.outlayer(x)
        return out

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
            # torch.nn.Linear(320, 50),
            torch.nn.Linear(1024, 40),
        )



# SlimNet模型 ConvBNReLU、DWSeparableConv、SSEBlock、SlimModule、SlimNet人脸属性检测
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

class DWSeparableConv(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.dwc = ConvBNReLU(inp, inp, kernel_size=3, groups=inp)
        self.pwc = ConvBNReLU(inp, oup, kernel_size=1)

    def forward(self, x):
        x = self.dwc(x)
        x = self.pwc(x)

        return x

class SSEBlock(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        out_channel = oup * 4
        self.pwc1 = ConvBNReLU(inp, oup, kernel_size=1)
        self.pwc2 = ConvBNReLU(oup, out_channel, kernel_size=1)
        self.dwc = DWSeparableConv(oup, out_channel)

    def forward(self, x):
        x = self.pwc1(x)
        out1 = self.pwc2(x)
        out2 = self.dwc(x)

        return torch.cat((out1, out2), 1)

class SlimModule(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        hidden_dim = oup * 4
        out_channel = oup * 3
        self.sse1 = SSEBlock(inp, oup)
        self.sse2 = SSEBlock(hidden_dim * 2, oup)
        self.dwc = DWSeparableConv(hidden_dim * 2, out_channel)
        self.conv = ConvBNReLU(inp, hidden_dim * 2, kernel_size=1)

    def forward(self, x):
        out = self.sse1(x)
        out += self.conv(x)
        out = self.sse2(out)
        out = self.dwc(out)

        return out

class SlimNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = ConvBNReLU(3, 96, kernel_size=7, stride=2)
        self.max_pool0 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.module1 = SlimModule(96, 16)
        self.module2 = SlimModule(48, 32)
        self.module3 = SlimModule(96, 48)
        self.module4 = SlimModule(144, 64)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.max_pool0(self.conv(x))
        x = self.max_pool1(self.module1(x))
        x = self.max_pool2(self.module2(x))
        x = self.max_pool3(self.module3(x))
        x = self.max_pool4(self.module4(x))
        x = self.gap(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)
        return x



