# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:54:54 2019

@author: asus
"""
import torch as t
from torchvision import transforms as T
from PIL import Image
from ResNet import *
from dic import *
from associate import *

def  pre(vir_path):
    
    checkpoint = t.load('./checkpoint3',map_location=t.device('cpu'))
    transforms = T.Compose([T.Resize((100,100)),T.Grayscale(),T.ToTensor()])
    model = ResNet(ResidualBlock)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img = transforms(Image.open(vir_path))
    img = t.unsqueeze(img,dim=0)

    if t.cuda.is_available():
        model = model.cuda()
        img =img.cuda()
    
    output = model(img)
    _, pred = t.max(output.data, 1)
   
    num = pred.item()
    char = list (char_dict.keys()) [list (char_dict.values()).index (num)]
    
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n您输入的汉字是：{}'.format(char))

    
    

    if char not in ass_dict.keys():
        print('我脑袋小，这个字还暂时没有联想到后面是什么……嘻嘻')
    else:
        print('联想：{}'.format(ass_dict[char]))
