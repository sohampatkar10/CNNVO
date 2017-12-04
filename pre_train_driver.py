#import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import autograd
import torch.nn.functional as F
from random import randint
import torch.nn as nn

import cv2
import random 
from random import uniform

from pre_train_class import *
from pretrainparser import *

# Defining the constants 
height = 32
width = 32
classes = 34 
epochs = 1 
batch_size = 48
 
model = TCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

ptp = PreTrainParser()
ptp = DataLoader(ptp, batch_size= 16, shuffle=True) 

def train_labeled_class(data):
	counter = 0
	for d in data: 
		x1 = d["img1"]
		x2 = d["img2"]
		# Converted to labels now 
		y  = d["tf"]
		#model.train()
			
		x1 = autograd.Variable(x1)
		x1= x1.view(16,1,28,28)
		x1 = x1.type(torch.FloatTensor)
		#print(x1.size())
		#x1 =x1.cuda()

		x2 = autograd.Variable(x2)
		x2= x2.view(16,1,28,28)
		x2 = x2.type(torch.FloatTensor)

		#x2=x2.cuda()

		y = autograd.Variable(y)
		#y = y.type(torch.FloatTensor)
		#y =y.cuda()
	 
		optimizer.zero_grad()
		#print("Shapes")
		#print(y)

		
		y_hat1, y_hat2, y_hat3 = model(x1, x2)
		print(y_hat3)

		y_hat1.type(torch.FloatTensor)
		y_hat2.type(torch.FloatTensor)
		y_hat3.type(torch.FloatTensor)
#        print(y[:,0])
		l1 = F.cross_entropy(y_hat1, y[:,0])
		l2 = F.cross_entropy(y_hat2, y[:,1])
		l3 = F.cross_entropy(y_hat3, y[:,2])

		loss= l1 + l2 + l3

		loss.backward()
		optimizer.step()
		print(loss.data[0])
		#print("l1", l1, "l2", l2, "l3", l3)

		counter = counter +1 
		#print(counter)
		#return 


for i in range(epochs):
	train_labeled_class(ptp)

#plt.plot(np.array(loss_itr), '-', linewidth=2)
torch.save(model.state_dict(),"./model") 














