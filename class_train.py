import torch
import torch.nn
from pre_train_class import *

class BCNN(torch.nn.Module):
    def __init__(self, pretrain_model = "model1_pretrain.pt"):
        super(BCNN, self).__init__()
        self.pretrain = PreTrain()
        self.pretrain.load_state_dict(torch.load(pretrain_model)) 
        self.conv3 = torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        
        self.conv4 = torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.max5 = torch.nn.MaxPool2d((3,3), stride=[2,2])
        
        
    def forward(self, x):
        x = self.pretrain(x)
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        x = self.max5(x)
        return x

class TCNN(torch.nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.drop = torch.nn.Dropout(p=0.5)
        self.linear1 = torch.nn.Linear(50*104*128, 500)
        self.linear2 = torch.nn.Linear(500, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 128*104*50)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.drop(x)
        return x
