import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
from random import randint
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch import autograd

import cv2
import random 
from random import uniform
from pre_train_class import *
from class_train import *

model1 = BCNN().cuda()
model2 = TCNN().cuda()

model1.load_state_dict(torch.load("./bcnn_model.pt"))
model2.load_state_dict(torch.load("./tcnn_model.pt"))

model1.eval()
model2.eval()

testset = DataParser('01')
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle = True)


correct = 0
total = 0

for counter, d in enumerate(trainloader,0):
	dtype = torch.cuda.FloatTensor
	x1 = d["img_l1"].type(dtype)
	x2 = d["img_l2"].type(dtype)
	yx = d["dx"].type(torch.cuda.LongTensor)
	yz = d["dz"].type(torch.cuda.LongTensor)
	yt = d["dth"].type(torch.cuda.LongTensor)
	 
	x1 = autograd.Variable(x1.cuda(), requires_grad= False)
	x2 = autograd.Variable(x2.cuda(), requires_grad= False)
	  
	yx = autograd.Variable(yx.cuda(), requires_grad= False)
	yz = autograd.Variable(yz.cuda(), requires_grad= False)
	yt = autograd.Variable(yt.cuda(), requires_grad= False)
	optimizer.zero_grad()

	f1 = model1(x1)
	f2 = model1(x2)

	f = torch.cat((f1, f2), 2)

	y_hat = model2(f) 
	y_hat.type(dtype)

    _, predicted = torch.max(outputs.data, 1)
    total += yx.size(0)
    correct += (predicted[:,0]==yx and predicted[:,1]==yz and predicted[:,2]==yt).sum()


print "correct = ", correct
print "total = ", total
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))