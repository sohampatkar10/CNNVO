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

model1 = BCNN().cuda()
model2 = TCNN().cuda()

model1.load_state_dict(torch.load("./bcnn_model.pt"))
model2.load_state_dict(torch.load("./tcnn_model.pt"))

model1.train()
model2.train()

optimizer = torch.optim.Adam((list(model1.parameters()) + list(model2.parameters())), lr=1e-4)

trainset = DataParser('01')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle = True)

epochs=10

ts = time.time() 
for e in range(epochs):
 print "epoch = ", e
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
  
  y_hx = y_hat[:,0:21]
  y_hz = y_hat[:, 21:42]
  y_ht = y_hat[:, 42:63]

  l1 = F.cross_entropy(y_hx, yx)
  l2 = F.cross_entropy(y_hz, yz)
  l3 = F.cross_entropy(y_ht, yt)
  loss = l1 + l2 + l3

  loss.backward()
  optimizer.step()
  print "i = ", counter, "loss = ", loss.data[0]

 torch.save(model1.state_dict(),"./bcnn_model.pt")
 torch.save(model2.state_dict(),"./tcnn_model.pt")


print "time taken = ", time.time()-ts 














