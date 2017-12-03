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

model = TCNN()
model.eval()

ptp = PreTrainParser(train=False)
ptp = DataLoader(ptp, batch_size= 16, shuffle=False) 

def test_labelled_class(data):

    correct = 0
    count = 0
    for d in data:
        count += 16
        x1 = d["img1"]
        x2 = d["img2"]
        # Converted to labels now 
        y  = d["tf"]
        #model.train()
            
        x1 = autograd.Variable(x1)
        x1= x1.view(16,1,32,32)
        x1 = x1.type(torch.FloatTensor)
        #x1 =x1.cuda()

        x2 = autograd.Variable(x2)
        x2= x2.view(16,1,32,32)
        x2 = x2.type(torch.FloatTensor)

        #x2=x2.cuda()

        y = autograd.Variable(y)

        y_hat1, y_hat2, y_hat3 = model(x1, x2)

        sm = torch.nn.Softmax()
        y1 = np.argmax(sm(y_hat1).data.numpy(), axis=1)
        y2 = np.argmax(sm(y_hat2).data.numpy(), axis=1)
        y3 = np.argmax(sm(y_hat3).data.numpy(), axis=1)

        for p in range(16):
            if(y1 == y_hat1 and y2 == y_hat2 and y3 == y_hat3):
                correct += 1

    print "accuracy = ", correct/float(count)