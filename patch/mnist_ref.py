import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/horan/Data/mnist_data', download=True, train=True, transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
batch_size=128, shuffle=True, drop_last=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/horan/Data/mnist_data', download=True, train=False,transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
 batch_size=128, shuffle=True, drop_last=True)

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
                
                
                return F.log_softmax(x)


# create classifier and optimizer objects
#clf = CNNClassifier()
#clf.cuda()


train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []
criterion1 = nn.CrossEntropyLoss()

def train(epoch):
    
    for batch_id, (data, label) in enumerate(train_loader):
        clf.train()
        data=data.cuda()
        label=label.cuda()
        opt.zero_grad()
        preds = clf(data)
        loss = torch.diag(preds[:,label])
        loss =  -torch.mean(loss)
        loss.backward()
        train_loss_history[-1].append(loss.item())
        opt.step()
        predind = preds.data.max(1)[1] 
        acc = predind.eq(label.data).cpu().float().mean() 
        train_acc_history[-1].append(acc)
        
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
            #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #    test_loss, correct, len(test_loader.dataset),
            #    accuracy))
            test_acc_history[-1].append(accuracy)
            test_loss_history[-1].append(test_loss)
            print("Test Loss: "+str(test_loss)+" Acc: "+str(accuracy))


for repeat in range(0, 1):
    clf = CNNClassifier()
    clf.cuda()
    opt = optim.SGD(clf.parameters(), lr=0.01, momentum=0.5)
    train_loss_history.append([])
    train_acc_history.append([])
    test_loss_history.append([])
    test_acc_history.append([])
    for epoch in range(0, 3):
        print("Epoch %d" % epoch)
        train(epoch)
    torch.save(clf.state_dict(),"RefModel.pth")
    #torch.save(clf.state_dict(), "LeNet-5")    
