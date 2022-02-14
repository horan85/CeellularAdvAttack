import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cv2

batch_size=128

# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/horan/Data/mnist_data', download=True, train=True, transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
batch_size=batch_size, shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/horan/Data/mnist_data', download=True, train=False,transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
 batch_size=batch_size, shuffle=False)

class CeNNLayer(nn.Module):
    def __init__(self, InDepth=1, OutDepth=1,TimeStep=0.1,IterNum=100):
        super(CeNNLayer, self).__init__()
        self.rescale= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.A= nn.Conv2d(OutDepth, OutDepth, kernel_size=3, padding=1)
        self.B= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.Z= nn.Parameter(torch.randn(OutDepth))
        #self.Z =self.Zsingle.view(1,OutDepth,1,1).repeat(16,1,28,28)
        self.TimeStep=0.1
        self.IterNum=10
        
    def NonLin(self,x,alpha=0.01):
    	y= torch.min(x,1+alpha*(x-1))
    	y= torch.max(y,-1+alpha*(y+1))
    	return y
           
    def forward(self, x):
        InputCoupling=self.B(x)
        Zreshaped=self.Z.view(1,InputCoupling.shape[1],1,1).repeat(InputCoupling.shape[0],1,InputCoupling.shape[2],InputCoupling.shape[3])
        InputCoupling=InputCoupling+Zreshaped
        x=self.rescale(x)
        for step in range(self.IterNum):
            Coupling=self.A(self.NonLin(x)) + InputCoupling
            x=x+self.TimeStep*(-x+Coupling)
        return self.NonLin(x)


class CellNN(nn.Module):
    def __init__(self):
        super(CellNN, self).__init__()
        self.Layer1= CeNNLayer(1,16)
        self.Layer2= CeNNLayer(16,32)
        self.Layer3= CeNNLayer(32,10)
        

    def forward(self, x):
    	x=self.Layer1(x)
    	x=self.Layer2(x)
    	x=self.Layer3(x)
    	return x


def SquaredDiff(NetOut,Labels):
	SquaredDiff=torch.mean(torch.square(NetOut-Labels))
	return  SquaredDiff
	
def SofMaxLoss(NetOut,Labels):
        preds=torch.mean(NetOut,[2,3])
        preds=torch.softmax(preds,-1)
        loss = torch.log(torch.diag(preds[:,Labels]))
        loss =  -torch.mean(loss)
        return loss,preds


    
clf = CellNN()
clf.cuda()
clf.load_state_dict(torch.load('CellModel.pth'))
clf.eval()

Noise=torch.nn.Parameter(torch.rand((1,1,28,28),device='cuda'))


opt = optim.SGD([Noise], lr=0.01, momentum=0.5)

(data, label) = next(iter(test_loader)) 

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
          loss,prob = SofMaxLoss(preds,AttackLabel)  
          loss.backward()
          opt.step()
          
          print(prob[0,AttackLabel].cpu().detach().numpy())
          Attacks[AttackInd,a]=prob[0,AttackLabel].cpu().detach().numpy()
          if a%10==0:
            prob=loss.item()
            #print("Attack Prob:"+str(np.exp(-prob)  ))
            #print("Attack Prob:"+str(np.exp(-prob)  ))
            #print("Attack Prob:"+str(probs[0,AttackLabel].cpu().detach().numpy()))
np.save('mnist_cell_attacks.npy',Attacks)


AttackInd=0
indata=data[AttackInd,0,:,:].view(1,1,28,28).cuda()
inlabel=label[AttackInd].cuda()
#print("Original Label: "+str(label.item()  ))
AttackLabel=3
for a in range(20000):
          opt.zero_grad()
          Clipped=torch.clip(Noise,-1.0,1.0) 
          
          
          preds = clf(indata+Clipped)     
          loss,prob = SofMaxLoss(preds,AttackLabel)  
          loss.backward()
          opt.step()
          
          print(prob[0,AttackLabel].cpu().detach().numpy())
          #print("Attack Prob:"+str(np.exp(-prob)  ))
            #print("Attack Prob:"+str(np.exp(-prob)  ))
            #print("Attack Prob:"+str(probs[0,AttackLabel].cpu().detach().numpy()))    

