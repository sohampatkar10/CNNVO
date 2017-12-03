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
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 96, kernel_size=11, stride=1, padding=0)
        self.max1  = torch.nn.MaxPool2d((3,3), stride=[1,1])
        
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0)
        self.max2 = torch.nn.MaxPool2d((3,3), stride=[1,1])

        #linear= torch.nn.Linear(13*13*256, 500)
        #drop = torch.nn.Dropout(p=0.5)
          
    def forward(self, x1, x2 ):
        # Features for the x1
        x1 = F.relu(self.conv1(x1))
        x1 = self.max1(x1)
        
        x1 = F.relu(self.conv2(x1))
        x1 = self.max2(x1)

        x1 = x1.view(-1, 10*10*256)

        #x1 = self.drop(self.linear(13*13*256,500))
        # Features for the x2

        x2 = F.relu(self.conv1(x2))
        x2 = self.max1(x2)
        
        x2 = F.relu(self.conv2(x2))
        x2 = self.max2(x2)

        x2 = x2.view(-1, 10*10*256)

        #x2 = self.drop(self.linear(13*13*256,500))

        #x12= torch.cat((x1,x2),1)
        #x =self.linearend(x12)
        return x1, x2

# Class for the Fine tuning
class Fine_Tune(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear= torch.nn.Linear(10*10*256, 500)
        self.drop = torch.nn.Dropout(p=0.5)
        
        
    def forward(self, x1 ,x2 ):
        x1 =x.view(-1, 10*10*256)
        x1 =F.relu(self.linear(x1))
        x1 =self.drop(x1)

        x2 =x.view(-1, 10*10*256)
        x2 =F.relu(self.linear(x2))
        x2 =self.drop(x2)

        return x1 , x2

# Class for the TCNN
class TCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear= torch.nn.Linear(13*13*256, 1000)
        self.pretrain = PreTrain()
        self.linear= torch.nn.Linear(2*10*10*256, 35)
        self.drop = torch.nn.Dropout(p=0.5)
        
        
    def forward(self, x1 , x2 ):
        x1, x2 = self.pretrain(x1, x2)
        x12= torch.cat((x1,x2),1)
        x = self.linear(x12)

        x = F.relu(x)
        x = self.drop(x)

        #print(x.size())
        x1 = x[:,0:7]
        x2 = x[:,7:14]
        x3 = x[:,14:35]  
        
        #print(x1)
        #_,pred_x1= torch.max(x1, 1)
        #print(pred_x1.size())
        #pred_x1 = pred_x1.data.numpy()
        #print(np.shape(pred_x1))

        #_, pred_x2= torch.max(x2, 1)
        #pred_x2 = pred_x2.data.numpy()
        #print(np.shape(pred_x1))

        #_, pred_x3= torch.max(x3, 1)
        #pred_x3 = pred_x3.data.numpy()
        #print(np.shape(pred_x1))

        #a= a.t()

        #b = torch.zeros(1,5)
        #b =b.t()

        #c = torch.ones(1,5)
        #c =c.t()

        #x  =torch.cat((x1,x2,x3),1)

        return x1, x2, x3



