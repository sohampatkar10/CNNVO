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
height = 32
width = 32
classes = 10
epochs = 1 
batch_size = 16
c =3

model1 = PreTrain()
model2 = PreTrain_TCNN()
optimizer = torch.optim.Adam([
                {'params': model1.parameters()},
                {'params': model2.parameters()}], lr=1e-4)
model1.train()
model2.train() 

#ptp = torchvision.datasets.CIFAR10('../',download=True, train=True)

ptp = PreTrainParser()
ptp = DataLoader(ptp, batch_size= 16, shuffle=True) 

def train_labeled_class(data):
	counter = 0
	for d in data: 
		x1 = d["img1"]
		y  = d["digit"]
			
		x1 = autograd.Variable(x1,requires_grad= False)
		x1= x1.view(batch_size, c, height, width)
		x1 = x1.type(torch.FloatTensor)

		y = autograd.Variable(y,requires_grad= False)

		optimizer.zero_grad()

		temp_yhat = model1(x1)
		y_hat = model2(temp_yhat)

		y_hat.type(torch.FloatTensor)

		loss= F.cross_entropy(y_hat, y)

		loss.backward()
		optimizer.step()
		print(loss.data[0])

		counter = counter +1  
		print(counter)


for i in range(epochs):
	train_labeled_class(ptp)

torch.save(model1.state_dict(),"./model_pretrain") 














