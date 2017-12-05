#import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
from random import randint
import torch.nn as nn
import torchvision.transforms as transforms

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

# model1 = PreTrain()
# model2 = PreTrain_TCNN()
# optimizer = torch.optim.Adam([
#                 {'params': model1.parameters()},
#                 {'params': model2.parameters()}], lr=1e-4)
# model1.train()
# model2.train()

model = SimpleLinearModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../', train=True,
                                        download=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

for counter, d in enumerate(trainloader,0):
	x1, y = d
	x1 = autograd.Variable(x1, requires_grad= False)

	x1 = x1.type(torch.FloatTensor)
	print (x1.size())
	# x1.view(4, 1, 3*32*32)
	y = autograd.Variable(y, requires_grad= False)
	optimizer.zero_grad()

	y_hat = model(x1)

	y_hat.type(torch.FloatTensor)

	loss= F.cross_entropy(y_hat, y)

	loss.backward()
	optimizer.step()
	if (counter % 2000 == 0):
		print("i = ", counter, "loss = ", loss.data[0])

# torch.save(model1.state_dict(),"./model_pretrain") 














