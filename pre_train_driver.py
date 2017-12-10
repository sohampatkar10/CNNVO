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
epochs = 50
batch_size = 16
c =3

model1 = PreTrain().cuda()
model2 = PreTrain_TCNN().cuda()

model1.load_state_dict(torch.load("./model1_pretrain.pt"))
model2.load_state_dict(torch.load("./model2_pretrain.pt"))

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

testset = torchvision.datasets.CIFAR10(root='../', train=False,
                                        download=True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                          shuffle=True, num_workers=2)

for e in range(epochs):
	print "epoch ", e
	model1.train()
	model2.train()
        try:
		for counter, d in enumerate(trainloader,0):
			x1, y = d
			print type(x1), type(y)
			x1 = autograd.Variable(x1.cuda(), requires_grad= False)
			y = autograd.Variable(y.cuda(), requires_grad= False)

			optimizer.zero_grad()
			y_hat = model2(model1(x1))

			y_hat.type(torch.cuda.FloatTensor)
			loss= F.cross_entropy(y_hat, y)

			loss.backward()
			optimizer.step()
			if counter%200 == 0:
				print("i = ", counter, "loss = ", loss.data[0])

		model1.eval()
		model2.eval()

		correct = 0
		total = 0
		for data in testloader:
   			images, labels = data
    			outputs = model2(model1(autograd.Variable(images.cuda())))
    			_, predicted = torch.max(outputs.data, 1)
    			total += labels.size(0)
    			correct += (predicted == labels.cuda()).sum()

		print "correct = ", correct
		print "total = ", total
		print('Accuracy of the network on the 10000 test images: %d %%' % (
    			100 * correct / float(total)))
	
	except KeyboardInterrupt:
		print "Keyboard interuppt, Saving models"
		torch.save(model1.state_dict(),"./model1_pretrain.pt")
		torch.save(model2.state_dict(),"./model2_pretrain.pt")
		print "Saved models"
		break

torch.save(model1.state_dict(),"./model1_pretrain.pt")
torch.save(model2.state_dict(),"./model2_pretrain.pt")













