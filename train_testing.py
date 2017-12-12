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
from dataparser import DataParser

model1 = BCNN().cuda()
model2 = TCNN().cuda()

model1.load_state_dict(torch.load("./bcnn_model.pt"))
model2.load_state_dict(torch.load("./tcnn_model.pt"))

model1.eval()
model2.eval()

testset = DataParser('04')
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle = True)

correct_x = 0
correct_z = 0
correct_t = 0
total = 0

for counter, d in enumerate(testloader,0):
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

	f1 = model1(x1)
	f2 = model1(x2)

	f = torch.cat((f1, f2), 2)

	y_hat = model2(f) 
	y_hat.type(dtype)

  	y_hx = y_hat[:,0:20]
  	y_hz = y_hat[:, 20:40]
  	y_ht = y_hat[:, 40:60]

    	_, predicted_x = torch.max(y_hx.data, 1)
    	_, predicted_z = torch.max(y_hz.data, 1)
    	_, predicted_t = torch.max(y_ht.data, 1)
    	total += yx.size(0)
 	correct_x += (predicted_x==yx.data).sum()
 	correct_z += (predicted_z==yz.data).sum()
 	correct_t += (predicted_t==yt.data).sum()

print "correct x", correct_x
print "correct z", correct_z
print "correct t", correct_t
print "total = ", total
print "x accuracy = ", correct_x/float(total)
print "z accuracy = ", correct_z/float(total)
print "t accuracy = ", correct_t/float(total)
