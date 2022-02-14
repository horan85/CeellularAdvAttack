import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

batch_size=128


# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('/home/horan/Data/cifar10_data', download=True, train=True, transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
batch_size=batch_size, shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('/home/horan/Data/cifar10_data', download=True, train=False,transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
 batch_size=batch_size, shuffle=False)



#definition of th network class
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        #two convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        #fully connected layer
        self.fc1 = nn.Linear(128*2*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
      
           
    def forward(self, x):
                #definition of the forward pass
                
                x=self.conv1(x)
                x=F.max_pool2d(x, 2 ,2)
                x = F.relu(x)
                
                x=self.conv2(x)
                x=F.max_pool2d(x, 2 ,2)
                x = F.relu(x)
                
                x=self.conv3(x)
                x=F.max_pool2d(x, 2 ,2)
                x = F.relu(x)
                           
                
                
                x= x.view(-1,128*2*2)
                
                x=self.fc1(x)
                x = F.relu(x)
                
                x=self.fc2(x)
                x = F.relu(x)
                
                x=self.fc3(x)
                
                
                return F.log_softmax(x,-1)


# create classifier and optimizer objects
#clf = CNNClassifier()
#clf.cuda()

criterion1 = nn.CrossEntropyLoss()

def train(epoch):
    #training for one epoch
    #reading data
    for batch_id, (data, label) in enumerate(train_loader):
        clf.train() #set the net to train mode
        data=data.cuda() #put data on gpu
        label=label.cuda()
        opt.zero_grad()
        preds = clf(data) #execute forward pass
        #calculate loss
        loss = torch.diag(preds[:,label])
        loss =  -torch.mean(loss)
        loss.backward() #backward pass
        opt.step() #update parameters
        #caclualte accuracy
        predind = preds.data.max(1)[1] 
        acc = predind.eq(label.data).cpu().float().mean() 
        
        if batch_id % 100 == 0:
            print("Train Loss: "+str(loss.item())+" Acc: "+str(acc.item()))

            #run independent test
            clf.eval() # set model in inference mode (need this because of dropout)
            test_loss = 0
            correct = 0
        
            for data, target in test_loader: 
                data=data.cuda()
                target=target.cuda()  
                with torch.no_grad():    
                   output = clf(data)
                   test_loss += F.nll_loss(output, target).item()
                   pred = output.data.max(1)[1] 
                   correct += pred.eq(target.data).cpu().sum()

            test_loss = test_loss
            test_loss /= len(test_loader) # loss function already averages over batch size
            accuracy =  correct.item() / len(test_loader.dataset)
            print("Test Loss: "+str(test_loss)+" Acc: "+str(accuracy))


        

criterion1 = nn.CrossEntropyLoss()

       
clf = CNNClassifier()
clf.cuda()
clf.load_state_dict(torch.load('CifarRefModel.pth'))
clf.eval()

Noise=torch.nn.Parameter(torch.zeros((1,3,32,32),device='cuda'))

opt = optim.SGD([Noise], lr=0.01, momentum=0.5)

(data, label) = next(iter(test_loader)) 
print(data.shape)

Attacks=np.zeros((100,5000))
for AttackInd in range(100):
      print(AttackInd)
      indata=data[AttackInd,:,:,:].view(1,3,32,32).cuda()
      inlabel=label[AttackInd].cuda()
      #print("Original Label: "+str(label.item()  ))
      AttackLabel=np.random.randint(0,9)
      if AttackLabel>=inlabel:
            AttackLabel+=1
      for a in range(5000):
          opt.zero_grad()
          NoisyImg=indata+Noise
          Clipped=torch.clip(NoisyImg,-1.0,1.0) 
          
          
          
          
          preds = clf(Clipped)     
          loss = torch.diag(preds[:,AttackLabel])
          loss = -torch.mean(loss)
          loss.backward()
          opt.step()
        
          
          prob=np.exp(-loss.item())
          Attacks[AttackInd,a]=prob
          #print(prob)
np.save('cifar_ref_attacks.npy',Attacks)
    
    
