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

"""
Driver script for testing the pre-trained model
"""

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='../', train=True,
                                        download=True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Load Models
model1 = PreTrain()
model2 = PreTrain_TCNN()
model1.load_state_dict(torch.load("./model1_pretrain.pt"))
model2.load_state_dict(torch.load("./model2_pretrain.pt"))
model1.cuda()
model2.cuda()

# Set mode to eval
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
    100 * correct / total))
