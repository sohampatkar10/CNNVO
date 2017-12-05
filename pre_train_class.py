import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
from random import randint
import torch.nn as nn

import cv2
import random 
from random import uniform


# Class for the Pre- Training 
class PreTrain(torch.nn.Module):
    def __init__(self):
        super(PreTrain,self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.max1  = torch.nn.MaxPool2d((3,3), stride=[1,1])

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.max2 = torch.nn.MaxPool2d((3,3), stride=[1,1])

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = F.relu(x1)
        x1 = self.max1(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.max2(x1)
        print ("size after conv2 = ", x1.size())
        x1 = x1.view(-1, 24*24*32)
        return x1 

class PreTrain_TCNN(torch.nn.Module):
    def __init__(self):
        super(PreTrain_TCNN, self).__init__()
        self.linear1 = torch.nn.Linear(24*24*32, 1000)
        self.linear2 = torch.nn.Linear(1000, 10)

    def forward(self, x1):
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        return x1

class SimpleLinearModel(torch.nn.Module):
  def __init__(self):
    super(SimpleLinearModel, self).__init__()
    self.linear1 = torch.nn.Linear(3*32*32, 1000)
    self.linear2 = torch.nn.Linear(1000,10)

  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))

    return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()