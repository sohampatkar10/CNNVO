#import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
from random import randint
import torch.nn as nn

import cv2
import random 
from random import uniform

from pre_train_class import *
from pretrainparser import *

# Defining the constants 
height = 28
width = 28
classes = 34 
epochs = 1 
batch_size = 48
 
model = TCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
model.train()

ptp = PreTrainParser()
ptp = DataLoader(ptp, batch_size= 16, shuffle=False) 

def train_labeled_class(data):

    for d in data: 
        x1 = d["img1"]
        x2 = d["img2"]
        # Converted to labels now 
        y  = d["tf"]
        #model.train()
            
        x1 = autograd.Variable(x1)
        x1= x1.view(16,1,28,28)
        x1 = x1.type(torch.FloatTensor)
        #x1 =x1.cuda()

        x2 = autograd.Variable(x2)
        x2= x2.view(16,1,28,28)
        x2 = x2.type(torch.FloatTensor)

        #x2=x2.cuda()

        y = autograd.Variable(y)
        #y = y.type(torch.FloatTensor)
        #y =y.cuda()
     
        optimizer.zero_grad()
        
        y_hat1, y_hat2, y_hat3 = model(x1, x2)
        #y_hat_= y_hat_.view(-1)

        #y1 = y[:,0]
        #y2 = y[:,1]
        #y3 = y[:,2]


        print(y[:,0])
        print(y[:,1])
        print(y[:,2])

        y_hat1.type(torch.FloatTensor)
        y_hat2.type(torch.FloatTensor)
        y_hat3.type(torch.FloatTensor)
#        print(y[:,0])
        l1 = F.cross_entropy(y_hat1, y[:,0])
        l2 = F.cross_entropy(y_hat2, y[:,1])
        l3 = F.cross_entropy(y_hat3, y[:,2])

        loss= l1 + l2 + l3

        loss.backward()
        optimizer.step()
        #print(loss.data[0])
        #return loss.data[0]


for i in range(epochs):
	train_labeled_class(ptp)

#plt.plot(np.array(loss_itr), '-', linewidth=2)
torch.save(model.state_dict(),"./model") 














