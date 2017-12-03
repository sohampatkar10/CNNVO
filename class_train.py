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
        self.linear = torch.nn.linear(2*NO, 60)
        
        
    def forward(self, x1, x2):
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = x.view(-1, NO)

        x2 = F.relu(self.conv1(x2))
        x2 = F.relu(self.conv2(x2))
        x2 = x.view(-1, NO)

        x12= torch.cat((x1,x2),1)
        x =self.linearend(x12)

        x = F.relu(self.linear(x))
        x = self.drop(x)
        return x

    # Feature vector of some 60 length
    def train_TCNN(data, batch_size, image_size):

        for d in data: 
            x1 = data["img_l"]
            x2 = data["img_r"]
            labels  = data["pose"]

            model.train()
                
            x1 = autograd.Variable(x1)
            x1= x1.view(batch_size,3,image_size,image_size)
            #x1 =x1.cuda()

            x2 = autograd.Variable(x2)
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

            loss = L1 + L2 + L3 
            loss.backward()
            optimizer.step()
            print(loss.data[0])
            return loss.data[0]

    def predict(data, batch_size, image_size):

        for d in data: 
            x1 = data["img_l"]
            x2 = data["img_r"]
            labels  = data["pose"]

            model.train()
                
            x1 = autograd.Variable(x1)
            x1= x1.view(batch_size,3,image_size,image_size)
            #x1 =x1.cuda()

            x2 = autograd.Variable(x2)
            x2= x2.view(batch_size,3,image_size,image_size)
            #x2=x2.cuda()

            y = autograd.Variable(labels)
            y = y.type(torch.FloatTensor)
            #y =y.cuda()
         
            optimizer.zero_grad()
            
            y_hat_= model(x1, x2)
            y_hat_= y_hat_.view(-1)

            #scored_data = model.forward(autograd.Variable(torch.from_numpy(rand_data.astype(np.float32))))
            _, int_prediction= torch.max(scored_data, 1)
            int_prediction = int_prediction.data.numpy()

            #y1 = pose[:,1]
            #y2 = pose[:,2]
            #y3 = pose[:,3]
            #l1 = F.cross_entropy(y_hat_, y)
            #l2 = F.cross_entropy(y_hat_, y)
            #l3 = F.cross_entropy(y_hat_, y)

            #loss = L1 + L2 + L3 
            #loss.backward()
            #optimizer.step()
            #print(loss.data[0])
            return predicted_labels