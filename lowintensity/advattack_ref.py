import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cv2


# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/horan/Data/mnist_data', download=True, train=True, transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
batch_size=128, shuffle=True, drop_last=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/horan/Data/mnist_data', download=True, train=False,transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
 batch_size=128, shuffle=False, drop_last=True)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
      
           
    def forward(self, x):
                x=self.conv1(x)
                x=F.max_pool2d(x, 2 ,2)
                x = F.relu(x)
                
                x=self.conv2(x)
                x=F.max_pool2d(x, 2 ,2)
                x = F.relu(x)
                
                
                x= x.view(-1,16*5*5)
                
                
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

       
clf = CNNClassifier()
clf.cuda()
clf.load_state_dict(torch.load('RefModel.pth'))
clf.eval()

Noise=torch.nn.Parameter(torch.zeros((1,1,28,28),device='cuda'))


opt = optim.SGD([Noise], lr=0.01, momentum=0.5)

(data, label) = next(iter(test_loader)) 
print(data.shape)
cv2.imwrite("sample.png",data[0,0,:,:].cpu().numpy()*255)

Attacks=np.zeros((100,5000))
for AttackInd in range(100):
      print(AttackInd)
      indata=data[AttackInd,0,:,:].view(1,1,28,28).cuda()
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
          print(prob)
          Attacks[AttackInd,a]=prob
          #print(prob)
np.save('mnist_ref_attacks.npy',Attacks)
    
    
