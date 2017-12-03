# Class for the Pre- Training 
class PreTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 96, kernel_size=11, stride=1, padding=0)
        self.max1  = torch.nn.MaxPool2d((3,3), stride=[1,1])
        
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=0)
        self.max2 = torch.nn.MaxPool2d((3,3), stride=[1,1])

        #self.linear= torch.nn.Linear(13*13*256, 500)
        #self.drop = torch.nn.Dropout(p=0.5)
          
    def forward(self, x1, x2 ):
        # Features for the x1
        x1 = F.relu(self.conv1(x1))
        x1 = self.max1(x1)
        
        x1 = F.relu(self.conv2(x1))
        x1 = self.max2(x1)

        x1 = x1.view(-1, 10*10*256)
        #x1 = self.drop(self.linear(13*13*256,500))
        # Features for the x2
        x2 = F.relu(self.conv1(x2))
        x2 = self.max1(x2)
        
        x2 = F.relu(self.conv2(x2))
        x2 = self.max2(x2)

        x2 = x2.view(-1, 10*10*256)
        #x2 = self.drop(self.linear(13*13*256,500))

        #x12= torch.cat((x1,x2),1)
        #x =self.linearend(x12)
        return x1, x2

# Class for the Fine tuning
class Fine_Tune(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear= torch.nn.Linear(10*10*256, 500)
        self.drop = torch.nn.Dropout(p=0.5)
        
        
    def forward(self, x1 ,x2 ):
        x1 =x.view(-1, 10*10*256)
        x1 =F.relu(self.linear(x1))
        x1 =self.drop(x1)

        x2 =x.view(-1, 10*10*256)
        x2 =F.relu(self.linear(x2))
        x2 =self.drop(x2)

        return x1 ,x2 

# Class for the TCNN
class TCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear= torch.nn.Linear(13*13*256, 1000)
        self.linear= torch.nn.Linear(1000, 34)
        self.drop = torch.nn.Dropout(p=0.5)
        
        
    def forward(self, x1 , x2 ):
        #self.linear= torch.nn.Linear(13*13*256, 1000)

        x12= torch.cat((x1,x2),1)
        x =self.linearend(x12)

        x =F.relu(self.linear(x))
        x =self.drop(x)
        return x

def train_labelled_class(data):

    for d in data: 
        x1 = data["img1"]
        x2 = data["img2"]
        y  = data["tf"]

        model.train()
            
        x1 = autograd.Variable(image1_left)
        #x1= x1.view(batch_size,3,image_size,image_size)
        #x1 =x1.cuda()

        x2 = autograd.Variable(image2_right)
        #x2= x2.view(batch_size,3,image_size,image_size)
        #x2=x2.cuda()

        y = autograd.Variable(y)
        y = y.type(torch.FloatTensor)
        #y =y.cuda()
     
        optimizer.zero_grad()
        
        y_hat_= model(x1, x2)
        y_hat_= y_hat_.view(-1)

        y1 = y[1]
        y2 = y[2]
        y3 = y[3]

        l1 = F.cross_entropy(y_hat_[0], y)
        l2 = F.cross_entropy(y_hat_[1], y)
        l3 = F.cross_entropy(y_hat_[2], y)
        loss= l1 + l2 + l3

        loss.backward()
        optimizer.step()
        print(loss.data[0])
        return loss.data[0]
