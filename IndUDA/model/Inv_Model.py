import torch.nn as nn
import torch
from . import utils as model_utils
import numpy as np


class InvBlock(nn.Module):
    def __init__(self, split_num, channel_num, subnet_constructor, clamp=1.):
        super(InvBlock, self).__init__()
        self.split_num1 = split_num
        self.split_num2 = channel_num-split_num
        self.clamp = clamp
        fdim = split_num
        self.F1 = subnet_constructor('Resnet', None, self.split_num2, self.split_num1)
        self.F2 = subnet_constructor('Resnet', None, self.split_num1, self.split_num2)
        self.F3 = subnet_constructor('Resnet', None, self.split_num1, self.split_num2)
        '''
        self.F = nn.Sequential(
            nn.Linear(fdim, fdim), nn.ReLU(inplace=False),
            nn.Linear(fdim, fdim)
            )
        self.G = nn.Sequential(
            nn.Linear(fdim, fdim), nn.ReLU(inplace=False),
            nn.Linear(fdim, fdim)
            )
        '''
    def forward(self, x, rev=False):
        # split
        x1, x2 = (x.narrow(1, 0, self.split_num1), x.narrow(1, self.split_num1, self.split_num2))
        if not rev:

            y1 = x1+self.F1(x2)
            self.y1_tmp = (torch.sigmoid(self.F2(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.y1_tmp))+self.F3(y1)
            '''
            y2 = x2 + self.G(x1)
            y1 = x1 + self.F(y2)
            '''
        else:

            self.y1_tmp = (torch.sigmoid(self.F2(x1)) * 2 - 1)
            y2 = (x2-self.F3(x1)).div(torch.exp(self.y1_tmp))
            y1 = x1 - self.F1(y2)
            '''
            y1 = x1 - self.F(x2)
            y2 = x2 - self.G(y1)
            '''
        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.y1_tmp)
        else:
            jac = -torch.sum(self.y1_tmp)

        return jac / x.shape[0]


class InvNet(nn.Module):
    def __init__(self, split_num, channel_num, subnet_constructor, block_num=5):
        super(InvNet, self).__init__()
        operations = []

        current_channel = split_num
        for j in range(block_num):
            b = InvBlock(current_channel, channel_num, subnet_constructor)
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.split_num1 = split_num
        self.split_num2 = channel_num-split_num

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)


        if cal_jacobian:
            return out
        else:
            return out

#subnet of inn
def subnet_constructor(net_structure, init, channel_in, channel_out):
    if net_structure == 'DBNet':
        if init != 'xavier':
            return DenseBlock(channel_in, channel_out, init)
        else:
            return DenseBlock(channel_in, channel_out)
    elif net_structure == 'Resnet':
        return ResBlock(channel_in, channel_out)
    else:
        return None

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out
