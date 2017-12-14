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

import matplotlib.pyplot as plt

model1 = BCNN()
model2 = TCNN()

model1.load_state_dict(torch.load("./bcnn_model.pt"))
model2.load_state_dict(torch.load("./tcnn_model.pt"))

model1 = model1.cuda()
model2 = model2.cuda()

model1.eval()
model2.eval()

testset = DataParser('04')
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle = False)

total = 0
err_x = 0
err_z = 0
err_t = 0

xs, zs, ts = [], [], []
xp, xp, tp = [], [], []

gt = np.eye(4, dtype=float)

for counter, d in enumerate(testloader,0):
  dtype = torch.cuda.FloatTensor
  x1 = d["img_l1"].type(dtype)
  x2 = d["img_l2"].type(dtype)
  yx = d["dx"].type(dtype)
  yz = d["dz"].type(dtype)
  yt = d["dth"].type(dtype)

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

  dx = y_hat[:,0].numpy()
  dz = y_hat[:, 1].numpy()
  dth = y_hat[:, 2].numpy()

  dg = np.array([[np.cos(dth),0.0,np.sin(dth), dx],
                 [0.0, 1.0, 0.0, 0.0],
                 [-np.sin(dth),0.0,np.cos(dth),dz],
                 [0.0, 0.0, 0.0, 1.0]])

  gt = np.dot(gt, dg)

  xp.append(gt[0,3])
  zp.append(gt[2,3])

  xs.append(d["x"].numpy())
  zs.append(d["z"].numpy())
  ts.append(d["t"].numpy())

plt.figure
plt.plot(ts, zs)
plt.plot(ts, zp)

plt.savefig('./zs_plot.png')
