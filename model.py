class PreTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.max1  = torch.nn.MaxPool2d((3,3), stride=[2,2])
        
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.max2 = torch.nn.MaxPool2d((3,3), stride=[2,2])
          
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        return x

class TCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear= torch.nn.Linear(13*13*256, 1024)
        self.drop = torch.nn.Dropout(p=0.5)
        
        
    def forward(self, x):
        x =x.view(-1, 13*13*256)
        x =F.relu(self.linear(x))
        x =self.drop(x)
        return x
    
class Train(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.max1  = torch.nn.MaxPool2d((3,3), stride=[2,2])
        
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.max2 = torch.nn.MaxPool2d((3,3), stride=[2,2])
        
        self.conv3 = torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        
        self.conv4 = torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.max5 = torch.nn.MaxPool2d((3,3), stride=[2,2])
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        
        x = F.relu(self.conv2(x))
        x = self.max2(x)
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.conv4(x))
        
        x = F.relu(self.conv5(x))
        x = self.max5(x)
        return x

class TCNN_Train(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=0)
        self.drop = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.linear(500, 60)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 500)
        x = F.relu(self.linear(x))
        x = self.drop(x)
        return x