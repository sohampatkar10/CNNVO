import matplotlib.pyplot as plt
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
height = 28
width = 28
classes = 34 
epochs = 1 
batch_size = 48
 
model = TCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

for i in range(epochs):
	train_labeled_class()















