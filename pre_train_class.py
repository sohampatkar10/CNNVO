# Class for the Pre- Training 
class PreTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.max1  = torch.nn.MaxPool2d((3,3), stride=[2,2])
        
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.max2 = torch.nn.MaxPool2d((3,3), stride=[2,2])

        #self.linear= torch.nn.Linear(13*13*256, 500)
        #self.drop = torch.nn.Dropout(p=0.5)
          
    def forward(self, x1, x2 ):
        # Features for the x1
        x1 = F.relu(self.conv1(x1))
        x1 = self.max1(x1)
        
        x1 = F.relu(self.conv2(x1))
        x1 = self.max2(x1)

        x1 = x1.view(-1, 13*13*256)
        #x1 = self.drop(self.linear(13*13*256,500))
        # Features for the x2
        x2 = F.relu(self.conv1(x2))
        x2 = self.max1(x2)
        
        x2 = F.relu(self.conv2(x2))
        x2 = self.max2(x2)

        x2 = x2.view(-1, 13*13*256)
        #x2 = self.drop(self.linear(13*13*256,500))

        #x12= torch.cat((x1,x2),1)
        #x =self.linearend(x12)
        return x1, x2

# Class for the Fine tuning
class Fine_Tune(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear= torch.nn.Linear(13*13*256, 500)
        self.drop = torch.nn.Dropout(p=0.5)
        
        
    def forward(self, x1 ,x2 ):
        x1 =x.view(-1, 13*13*256)
        x1 =F.relu(self.linear(x1))
        x1 =self.drop(x1)

        x2 =x.view(-1, 13*13*256)
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

def train_labelled_class(image_left, image_right, labels, batch_size, image_size, time_stamp):

    for d in data: 
        x1 = data["img_l"]
        x2 = data["img_r"]
        y  = data["pose"]

        model.train()
            
        x1 = autograd.Variable(image1_left)
        x1= x1.view(batch_size,3,image_size,image_size)
        #x1 =x1.cuda()

        x2 = autograd.Variable(image2_right)
        x2= x2.view(batch_size,3,image_size,image_size)
        #x2=x2.cuda()

        y = autograd.Variable(labels)
        y = y.type(torch.FloatTensor)
        #y =y.cuda()
     
        optimizer.zero_grad()
        
        y_hat_= model(x1, x2)
        y_hat_= y_hat_.view(-1)
        y1 = pose[:,1]
        y2 = pose[:,2]
        y3 = pose[:,3]
        l1 = F.cross_entropy(y_hat_, y)
        l2 = F.cross_entropy(y_hat_, y)
        l3 = F.cross_entropy(y_hat_, y)
        loss.backward()
        optimizer.step()
        print(loss.data[0])
        return loss.data[0]
