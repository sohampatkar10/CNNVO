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
"""
Driver script for training the model
"""

# Instiate the models
model1 = BCNN()
model2 = TCNN()

# Load models from last saved state
model1.load_state_dict(torch.load("./bcnn_model.pt"))
model2.load_state_dict(torch.load("./tcnn_model.pt"))
model1.cuda()
model2.cuda()

# Set optimizer as Adam
optimizer = torch.optim.Adam(
    (list(model1.parameters()) + list(model2.parameters())), lr=1e-4)

# Load Training data
trainset = DataParser('01')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)

# Load testing data
testset = DataParser('04')
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

# criterion is MSELoss
criterion = nn.MSELoss().cuda()

epochs = 15
i = 0

for e in range(epochs):
    model1.train()
    model2.train()

    ts = time.time()
    for data in trainloader:
        i += 1

        # Get images and labels from data
        i1 = data["img_l1"]
        i2 = data["img_l2"]
        yx = data["dx"]
        yz = data["dz"]
        yt = data["dth"]

        # Convert to torch Variable
        i1 = Variable(i1)
        i2 = Variable(i2)
        yx = Variable(yx)
        yz = Variable(yz)
        yt = Variable(yt)

        # Give inputs to BCNN
        f1 = model1(i1)
        f2 = model1(i2)

        # Concatenate and give input to TCNN
        yh = model2(torch.cat((f1, f2), 2))

        # Compute loss
        l1 = criterion(yh[:, 0], yx)
        l2 = criterion(yh[:, 1], yz)
        l3 = criterion(yh[:, 2], yt)
        loss = l1+l2+l3

        # Backpropogation and weights update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print "time for epoch = ", time.time()-ts
