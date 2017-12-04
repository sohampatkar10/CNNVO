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
        self.norm1 = torch.nn.BatchNorm2d(96)
        self.max1  = torch.nn.MaxPool2d((3,3), stride=[1,1])

        
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0)
        self.norm2 = torch.nn.BatchNorm2d(256)
        self.max2 = torch.nn.MaxPool2d((3,3), stride=[1,1])

          
    def forward(self, x1, x2 ):
       

        x1 =self.conv1(x1)
        x1 = self.norm1(F.relu(x1))
        x1 = self.max1(x1)
        x1 = self.norm2(F.relu(self.conv2(x1)))
        x1 = self.max2(x1)
        x1 = x1.view(-1, 10*10*256)

        x2 = self.conv1(x2)
        x2 = F.relu(x2)
        x2 = self.max1(x2)
        x2 = F.relu(self.conv2(x2))
        x2 = self.max2(x2)
        x2 = x2.view(-1, 10*10*256)
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

        self.X= torch.nn.Linear(35, 7)
        self.Y= torch.nn.Linear(35, 7)
        self.T= torch.nn.Linear(35, 21)
        
        
    def forward(self, x1 , x2 ):
        x1, x2 = self.pretrain(x1, x2)

        x12= torch.cat((x1,x2),1)
        counter =0 
        for i in range (2*10*10*256):
            if x12[:,i].data[0]>0.1: 
                counter = counter +1 
        #print(counter)
        #print(x12)
        x = self.linear(x12)
        #print(x)

        x = F.relu(x)
        x = self.drop(x)
        #print(x.size())
        x1 = F.relu(self.X(x))
        x2 = F.relu(self.Y(x))
        x3 = F.relu(self.T(x))  
        
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


        return x1, x2, x3

class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 96, kernel_size=11, stride=1, padding=0)
        self.max1  = torch.nn.MaxPool2d((3,3), stride=[1,1])

        
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0)
        self.max2 = torch.nn.MaxPool2d((3,3), stride=[1,1])

        self.linear1 = torch.nn.Linear(10*10*256, 1000)
        self.linear2 = torch.nn.Linear(1000, 7)

        self.drop = torch.nn.Dropout(p=0.5)

        #self.X= torch.nn.Linear(35, 7)
        #self.Y= torch.nn.Linear(35, 7)
        #self.T= torch.nn.Linear(35, 21)

        b = torch.rand(35)
        self.b = autograd.Variable(b,requires_grad= True)

          
    def forward(self, x1, x2 ):
        x1 =self.conv1(x1)
        x1 = F.relu(x1)
        x1 = self.max1(x1)
        x1 = F.relu(self.conv2(x1))
        x1 = self.max2(x1)
        x1 = x1.view(-1, 10*10*256)
        x1 = self.linear1(x1)
        x1 = self.linear2(x1)

        # x2 = self.conv1(x2)
        # x2 = F.relu(x2)
        # x2 = self.max1(x2)
        # x2 = F.relu(self.conv2(x2))
        # x2 = self.max2(x2)
        # x2 = x2.view(-1, 10*10*256)
        # x2 = self.linear1(x2)

        # x12= torch.cat((x1,x2),1)

        # x = self.linear2(x12)

        # x = F.relu(x+ self.b)
        # x = self.drop(x)
        # print("Bias",self.b.data[0])

        #x_x = F.relu(self.X(x)) 
        #print("Printing x wieghts",x_x)
        #y_y = F.relu(self.Y(x))
        #z_z = F.relu(self.T(x))

        return x1


