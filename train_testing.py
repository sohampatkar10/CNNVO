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

"""
Script for testing the trained model
"""

# Instantiate models
model1 = BCNN()
model2 = TCNN()

# Load models from files
model1.load_state_dict(torch.load("./bcnn_model.pt"))
model2.load_state_dict(torch.load("./tcnn_model.pt"))
model1 = model1.cuda()
model2 = model2.cuda()

# Set to eval mode
model1.eval()
model2.eval()

# Load Data from testing set
testset = DataParser('04')
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

total = 0
err_x = 0
err_z = 0
err_t = 0

# Run for all testing data
for counter, d in enumerate(testloader, 0):
    dtype = torch.cuda.FloatTensor
    x1 = d["img_l1"].type(dtype)
    x2 = d["img_l2"].type(dtype)
    yx = d["dx"].type(dtype)
    yz = d["dz"].type(dtype)
    yt = d["dth"].type(dtype)

    x1 = autograd.Variable(x1.cuda(), requires_grad=False)
    x2 = autograd.Variable(x2.cuda(), requires_grad=False)

    yx = autograd.Variable(yx.cuda(), requires_grad=False)
    yz = autograd.Variable(yz.cuda(), requires_grad=False)
    yt = autograd.Variable(yt.cuda(), requires_grad=False)

    f1 = model1(x1)
    f2 = model1(x2)

    f = torch.cat((f1, f2), 2)

    y_hat = model2(f)
    y_hat.type(dtype)

    y_hx = y_hat[:, 0]
    y_hz = y_hat[:, 1]
    y_ht = y_hat[:, 2]

    total += yx.size(0)
    err_x += abs((yx.data-y_hx.data).cpu().numpy()).sum()
    err_z += abs((yz.data-y_hz.data).cpu().numpy()).sum()
    err_t += abs((yt.data-y_ht.data).cpu().numpy()).sum()

print "av err x = ", err_x/float(total)
print "av err z = ", err_z/float(total)
print "av err t = ", err_t/float(total)
