#import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
from random import randint
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

import cv2
import random 
from random import uniform

from pre_train_class import *
import time

# Defining the constants 
height = 32
width = 32
classes = 10
epochs = 1 
batch_size = 16
c =3

model1 = PreTrain()
model2 = PreTrain_TCNN()

model1.train()
model2.train()
optimizer = torch.optim.Adam((list(model1.parameters()) + list(model2.parameters())), lr=1e-4)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../', train=True,
                                        download=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

for counter, d in enumerate(trainloader,0):
	x1, y = d
	x1 = autograd.Variable(x1, requires_grad= False)

	x1 = x1.type(torch.cuda.FloatTensor)
	# x1 = x1.view(-1, 3*32*32)
	y = autograd.Variable(y, requires_grad= False)
	optimizer.zero_grad()

	y_hat = model2(model1(x1))

	y_hat.type(torch.cuda.FloatTensor)

	loss= F.cross_entropy(y_hat, y)

	loss.backward()
	optimizer.step()
	print("i = ", counter, "loss = ", loss.data[0])

torch.save(model1.state_dict(),"./model1_pretrain.pt")
torch.save(model2.state_dict(),"./model2_pretrain.pt")














