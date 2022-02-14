import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cv2

batch_size=32

# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('/home/horan/Data/cifar10_data', download=True, train=True, transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
batch_size=batch_size, shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('/home/horan/Data/cifar10_data', download=True, train=False,transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
 batch_size=batch_size, shuffle=True)

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
        self.Layer1= CeNNLayer(3,32)
        self.Layer2= CeNNLayer(32,64)
        self.Layer3= CeNNLayer(64,128)
        self.Layer4= CeNNLayer(64,128)
        self.Layer5= CeNNLayer(128,256)
        self.Layer6= CeNNLayer(128,10)
        
        

    def forward(self, x):
    	x=self.Layer1(x)
    	x=self.Layer2(x)
    	x=self.Layer3(x)
    	#x=self.Layer4(x)
    	#x=self.Layer5(x)
    	x=self.Layer6(x)
    	return x


def SquaredDiff(NetOut,Labels):
	SquaredDiff=torch.mean(torch.square(NetOut-Labels))
	return  SquaredDiff
	
def SofMaxLoss(NetOut,Labels):
        preds=torch.mean(NetOut,[2,3])
        preds=torch.softmax(preds,-1)
        loss = torch.log(torch.diag(preds[:,Labels]))
        loss =  -torch.mean(loss)
        return loss
        
def train(epoch):
    for batch_id, (data, label) in enumerate(train_loader):
        clf.train()
        data=data.cuda()
        label=label.cuda()
        opt.zero_grad()
        preds = clf(data)
        
        #one_hot = torch.zeros(preds.shape[0], preds.shape[1]).cuda()
        #one_hot[torch.arange(preds.shape[0]), label] = 1
        #ImgLabels=one_hot.view(preds.shape[0], preds.shape[1],1,1).repeat(1,1,preds.shape[2],preds.shape[3])
        #ImgLabels=(2*ImgLabels-1.0)
        
        
        #loss = SquaredDiff(preds,ImgLabels)
        loss = SofMaxLoss(preds,label)
        
        loss.backward()
        opt.step()
        predind = torch.sum(preds, [2,3])
        predind = predind.data.max(1)[1] 
        acc = predind.eq(label.data).cpu().float().mean() 

        if batch_id % 100 == 0:
            
            print("Train Loss: "+str(loss.item())+" Acc: "+str(acc.item()))
 
            #run independent test
            clf.eval() # set model in inference mode (need this because of dropout)
            test_loss = 0
            correct = 0
            SampleNum=0
            for data, target in test_loader: 
                if data.shape[0]==batch_size:
                        datab=data.cuda()
                        label=target.cuda()  
                        with torch.no_grad():    
                           output = clf(datab)
                           pred = torch.sum(output, [2,3]).data.max(1)[1] 
                           correct += pred.eq(label.data).cpu().sum()
                        SampleNum+=data.shape[0]
            accuracy =  correct.item() / SampleNum
            print("Test Acc: "+str(accuracy))
            ImgNumX=4
            ImgNumY=8
            ImgSize=32
            ImgArray=np.zeros((2*ImgNumX*ImgSize,ImgNumY*ImgSize,3))
            for i in range(ImgNumX):
                for j in range(ImgNumY):
                    Img=datab[j*ImgNumX+i,:,:,:].cpu().detach().numpy()
                    Img-=np.amin(Img) 
                    Img/=np.amax(Img)         
                    ImgArray[2*i*ImgSize:(2*i+1)*ImgSize,j*ImgSize:(j+1)*ImgSize,0]= Img[2,:,:]*255 
                    ImgArray[2*i*ImgSize:(2*i+1)*ImgSize,j*ImgSize:(j+1)*ImgSize,1]= Img[1,:,:]*255 
                    ImgArray[2*i*ImgSize:(2*i+1)*ImgSize,j*ImgSize:(j+1)*ImgSize,2]= Img[0,:,:]*255   
                    Img=output[j*ImgNumX+i,label[j*ImgNumX+i],:,:].cpu().detach().numpy()
                    Img-=np.amin(Img) 
                    Img/=np.amax(Img)   
                    ImgArray[(2*i+1)*ImgSize:(2*i+2)*ImgSize,(j)*ImgSize:(j+1)*ImgSize,0]=Img *255  
                    ImgArray[(2*i+1)*ImgSize:(2*i+2)*ImgSize,(j)*ImgSize:(j+1)*ImgSize,1]=Img *255  
                    ImgArray[(2*i+1)*ImgSize:(2*i+2)*ImgSize,(j)*ImgSize:(j+1)*ImgSize,2]=Img *255             
            cv2.imwrite("cifarimgs.png",ImgArray)
            
clf = CellNN()
#for p in clf.parameters():
#                print(p.shape)
clf.cuda()
opt = optim.Adam(clf.parameters(), lr=0.001)
for epoch in range(0, 20):
        print("Epoch %d" % epoch)
        train(epoch)
        torch.save(clf.state_dict(),"CifarCellModel.pth")  
