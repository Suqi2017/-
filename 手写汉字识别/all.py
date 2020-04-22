# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:32:30 2019

@author: asus
"""
import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import argparse
from ResNet import *

class Config(object):
    image_size=100
    lr = 0.001
    num_class = 100
    batch_size = 64
    epoch = 30
    txt_path = './new_txt_path.txt'
    root ='./data/train_P1'
    log_path ='./checkpoint'
    restore = False
'''
class CNN(nn.Module)
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        #self.global_avg = nn.AveragePool((1,1))
        self.fc4 = nn.Linear(512,3755)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc4(x)
        return x 
'''        

    


class MyDataset(Dataset):
    def __init__(self,txt_path,num_class,transforms=None):
        super(MyDataset,self).__init__()
        images = []
        labels = []
        with open(txt_path,'r') as f:
            for line in f:
                #print(int(line.split(os.sep)[-2]))
                line = line.strip('\n')
                if int(line.split('/')[-2]) >= num_class:
                    continue
                
                images.append(line)
                labels.append(int(line.split('/')[-2]))
                
                
        self.images = images
        self.labels = labels
        self.transforms =transforms
    
    def __getitem__(self,index):
        image = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image,label
    
    def __len__(self):
        return len(self.labels)


def train():
    Configs = Config()
    
    transforms = T.Compose([T.Resize((Configs.image_size,Configs.image_size)),T.Grayscale(),T.ToTensor(),T.Normalize(mean=[0.5],std=[0.5])])
    train_set = MyDataset(txt_path=Configs.txt_path,num_class=Configs.num_class,transforms=transforms)
    
    train_loader = DataLoader(train_set, batch_size=Configs.batch_size, shuffle=True)
    # 指定运行处理器
    
    model = ResNet(ResidualBlock)
    if t.cuda.is_available():
        model=model.cuda()
        
    model.train()
    
    #loss Func
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=Configs.lr)
    
    # 是否回复训练
    if Configs.restore:
        checkpoint = t.load(Configs.log_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        epoch = checkpoint['epoch']
    else:
        
        loss = 0.0
        epoch = 0
    
    
    while epoch < Configs.epoch:
        running_loss = 0.0
        
        for i,data in enumerate(train_loader):
            inputs,labels =data[0].cuda(),data[1].cuda()
            
            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:  # every 200 steps
                print('epoch %5d: batch: %5d, loss: %f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
            
        if epoch % 10 == 9:
            print('Save checkpoint...')
            t.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                       Configs.log_path)
        epoch += 1
    print('Finish training')
    
        
train()
    
