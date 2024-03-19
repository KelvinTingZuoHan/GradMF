import copy

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch


class Grad():#optim.Optimizer
    def __init__(self, optimizer ):
        """包装优化器
        """
        self._optim = optimizer
        return

    def optimizer(self):
        return  self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad()

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self,objectives_i,objectives_j):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''
        # print("2")
        
        grads_j, shapes_j = self.pack_grad(objectives_j)
        grads_i, shapes_i = self.pack_grad(objectives_i)
            # print("3")
        grad = self.comput_gradients_1(grads_i,grads_j )
        grad = self.unflatten_grad(grad, shapes_i[0])
        self._set_grad(grad)

        return


    def comput_gradients_1(self,grad_i,grad_j):

        eps = 1e-5

        
        for i in grad_i:
            mo_i = i.norm()+eps

        for j in grad_j:
            grad_j = j

        for i in grad_i:
            grad_i = i

        # mo_j = np.linalg.norm(grad_j)
        # print('mo1=',mo_i,'\nmo2=',mo_j)
        # cos_angle = float(grad_i.dot(grad_j)) / float(mo_i * mo_j)
        # angle = np.arccos(cos_angle)

    
        moj_cos = float(grad_j.dot(grad_i)) / float(mo_i) # 向量j的模乘以ij的cos值

        gradj_i = moj_cos*(1.0/mo_i)*grad_i
        print(moj_cos,gradj_i)
        gradj_x = grad_j-gradj_i

        grad_i_new = grad_i+gradj_x
        # x_i_g = [0, i_g[0]]
        # y_i_g = [0, i_g[1]]
        # plt.plot(x_i_g, y_i_g, linewidth=2, color='r')
        # print(grad_i_new)
        
        return grad_i_new


    def comput_gradients_2(self,grad_i,grad_j):
        eps = 1e-5
        for j in grad_j:
            mo_j = j.norm()+eps
       

        for j in grad_j:
            grad_j = j

        for i in grad_i:
            grad_i = i
        moi_cos = float(grad_i.dot(grad_j)) / float(mo_j)

        gradi_j = moi_cos*(1.0/mo_j)*grad_i
        return gradi_j

    # def Judgment_gradient(self,grad_i,grad_j):
    #
    #     # grad_i = np.array(grad_i)
    #     # grad_j = np.array(grad_j)
    #     grad_i = copy.deepcopy(grad_i)
    #     grad_j = copy.deepcopy(grad_j)
    #
    #     print(grad_i,grad_j)
    #     start_index = 0
    #     if grad_i.ndim>1  : #判断是否为矩阵，大于1则为dimension维矩阵
    #
    #         dimension = grad_i.ndim#计算维度
    #         print(dimension)
    #         print (grad_i.shape)
    #         original_shape = grad_i.shape
    #         # grad_i = np.squeeze(grad_i)
    #         # grad_j = np.squeeze(grad_j)#降维至二维
    #         grad_i = np.ravel(grad_i)
    #         grad_j = np.ravel(grad_j)#降维至二维
    #
    #         print(grad_i.shape)
    #     # for x in range(grad_i.shape[0]):  # 计算每一行单独的更新后的梯度值，从第零行开始
    #     #     print(x)
    #
    #         # grad_i_x = grad_i[x]
    #         # grad_i_x.astype(np.float64)
    #         print(grad_i)
    #         # grad_j_x = grad_j[x]
    #         # grad_j_x.astype(np.float64)
    #
    #         grad_i_new = self.comput_gradients(grad_i, grad_j)
    #         grad_i_new = grad_i_new.reshape(original_shape)
    #         print(grad_i_new.shape)
    #         return grad_i_new
    #     else:
    #         return self.comput_gradients(grad_i=grad_i,grad_j=grad_j)

    def unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad


    def pack_grad(self, objectives):

        grads, shapes = [], []
        # print(objectives.shape)
        self._optim.zero_grad()
        
        # print(objectives)
        objectives.backward(retain_graph=True)
        # print("after backward")
        grad, shape = self.retrieve_grad()
        grads.append(self.flatten_grad(grad, shape))
        shapes.append(shape)
        """for obj in objectives:
                self._optim.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                print(obj)
                obj.backward(retain_graph=True)
                print("after backward")
                grad, shape = self.retrieve_grad()
                grads.append(self.flatten_grad(grad, shape))
                shapes.append(shape)"""
        # print(type(grad))
        return grads, shapes


    def retrieve_grad(self):
        grad, shape= [], [],
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())

            return grad, shape,


    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:

                # if p.grad is None: continue
                p.grad = grads[idx]

                idx += 1
        return

    def flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad



class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)

class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)




if  __name__ == '__main__':

    # x = np.load("..\ST-GCN_layer1_NTU60-49HC.npy")
    # # print(np.split(x, 2, axis = 0))
    # print(x.shape)
    # y = x
    # i = np.array([[[0,1],[0,1],[1,1]],
    #               [[0,1],[0,1],[1,1]]])
    # j = np.array([[[1,1],[1,1],[0,1]],
    #               [[1,1],[1,1],[0,1]]])
    # grad = Grad(optim.Adam)
    # i_1 = grad.comput_gradients_2(grad_i=i,grad_j=j)
    # print(i_1)


    # torch.manual_seed(4)
    # x, y = torch.randn(2, 3), torch.randn(2, 4)
    # net = TestNet()
    # y_pred = net(x)
    # Grad_adam = Grad(optim.Adam(net.parameters()))
    # Grad_adam.zero_grad()
    # loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    # loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)
    #
    #
    #
    #
    # Grad_adam.pc_backward([loss1], [loss2])
    # for p in net.parameters():
    #     print(p.grad)

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    adam = Grad(optim.Adam(net.parameters()))
    adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)
    # print(type(loss1))

    adam.pc_backward(loss1, loss2)
    # for p in net.parameters():
        # print(p.grad)





