# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:19:55 2019

@author: asus
"""

import torch as t
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
'''
继承Dataset 并且重新写 __getitem__ 和 __len__ 方法

'''

    
class MyDataset(Dataset):
    def __init__(self,txt_path,num_class,transforms=None):
        super(MyDataset,self).__init__()
        images = []
        labels = []
        with open(txt_path,'r') as f:
            for line in f:
                if int(line.split('/')[-2])>=num_class:
                    break
                line = line.strip('\n')
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



