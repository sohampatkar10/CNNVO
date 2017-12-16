import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle = True)

testset = DataParser('04')
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle = True)

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(list(model1.parameters())+list(model2.parameters()), lr=1e-6)

epochs=15
i = 0

for e in range(epochs):
  model1.train()
  model2.train()

  ts = time.time()
  for data in trainloader:
    i += 1

    i1 = data["img_l1"]
    i2 = data["img_l2"]
    yx = data["dx"]
    yz = data["dz"]
    yt = data["dth"]

    i1 = Variable(i1)
    i2 = Variable(i2)
    yx = Variable(yx)
    yz = Variable(yz)
    yt = Variable(yt)
    
    f1 = model1(i1)
    f2 = model1(i2)

    yh = model2(torch.cat((f1, f2), 2))

    l1 = criterion(yh[:,0],yx)
    l2 = criterion(yh[:,1],yz)
    l3 = criterion(yh[:,2],yt)
    loss = l1+l2+l3

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
 
  print "time for epoch = ", time.time()-ts
