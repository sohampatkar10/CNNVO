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

model1 = PreTrain()
model2 = PreTrain_TCNN()

model1.eval()
model2.eval() 

ptp = PreTrainParser(train=False)
ptp = DataLoader(ptp, batch_size= 16, shuffle=False) 

def test_labelled_class(data):

    correct = 0
    count = 0
    for d in data:
        count += 16
        x1 = d["img1"]
        # Converted to labels now 
        y  = d["digit"]

        x1 = autograd.Variable(x1)
        x1= x1.view(16,1,32,32)
        x1 = x1.type(torch.FloatTensor)

        y = autograd.Variable(y)

        y_hat = model2(model1(x1))

        sm = torch.nn.Softmax()
        y1 = np.argmax(sm(y_hat1).data.numpy(), axis=1)

        for p in range(16):
            if y1 == y_hat1:
                correct += 1

    print "accuracy = ", correct/float(count)