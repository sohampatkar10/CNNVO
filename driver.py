import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

from class_train import BCNN, TCNN
from dataparser import DataParser
from pre_train_class import *
import time

model1 = BCNN()
model2 = TCNN()

model1.load_state_dict(torch.load("./bcnn_model.pt"))
model2.load_state_dict(torch.load("./tcnn_model.pt"))

model1.cuda()
model2.cuda()

optimizer = torch.optim.Adam((list(model1.parameters()) + list(model2.parameters())), lr=1e-4)

trainset = DataParser('01')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle = True)

testset = DataParser('04')
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle = True)

epochs=15

for e in range(epochs):
 model1.train()
 model2.train()
 print "epoch = ", e
 for counter, d in enumerate(trainloader,0):
  ts = time.time()
  dtype = torch.cuda.FloatTensor
  x1 = d["img_l1"].type(dtype)
  x2 = d["img_l2"].type(dtype)
  yx = d["dx"].type(dtype)
  yz = d["dz"].type(dtype)
  yt = d["dth"].type(dtype)
  print "time for data loading = ", time.time()-ts 
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
  y_hx = y_hat[:, 0]
  y_hz = y_hat[:, 1]
  y_ht = y_hat[:, 2]

  ts = time.time()
  l1 = F.mse_loss(y_hx, yx)
  l2 = F.mse_loss(y_hz, yz)
  l3 = F.mse_loss(y_ht, yt)
  loss = l1 + l2 + l3
  print "time for loss comp = ", time.time()-ts
  #loss = l2

  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
 
 print "e = ", e, "loss = ", loss.data[0] 
 torch.save(model1.state_dict(),"./bcnn_model.pt")
 torch.save(model2.state_dict(),"./tcnn_model.pt")

 model1.eval()
 model2.eval()

 total = 0
 err_x = 0
 err_z = 0
 err_t = 0
 
 if e%4==0:
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

        y_hx = y_hat[:,0]
        y_hz = y_hat[:, 1]
        y_ht = y_hat[:, 2]

        total += yx.size(0)
        err_x += abs((yx.data-y_hx.data).cpu().numpy()).sum()
        err_z += abs((yz.data-y_hz.data).cpu().numpy()).sum()
        err_t += abs((yt.data-y_ht.data).cpu().numpy()).sum()

  print "av err x = ", err_x/float(total)
  print "av err z = ", err_z/float(total)
  print "av err t = ", err_t/float(total)










