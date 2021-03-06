import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
from random import randint
import torch.nn as nn

import cv2
import random
from random import uniform


class PreTrain(torch.nn.Module):
    """
    Class for the Pre- Training
    """

    def __init__(self):
        super(PreTrain, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            3, 96, kernel_size=11, stride=1, padding=0)
        self.max1 = torch.nn.MaxPool2d((3, 3), stride=[1, 1])

        self.conv2 = torch.nn.Conv2d(
            96, 256, kernel_size=5, stride=1, padding=0)
        self.max2 = torch.nn.MaxPool2d((3, 3), stride=[1, 1])

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = F.relu(x1)
        x1 = self.max1(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.max2(x1)
        return x1


class PreTrain_TCNN(torch.nn.Module):
    def __init__(self):
        super(PreTrain_TCNN, self).__init__()
        self.linear1 = torch.nn.Linear(14*14*256, 1000)
        self.linear2 = torch.nn.Linear(1000, 10)

    def forward(self, x1):
        x1 = x1.view(-1, 14*14*256)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        return x1
