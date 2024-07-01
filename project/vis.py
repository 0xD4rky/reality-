## INSTALLING AND LOADING ALL THE NECESSARY LIBRARIES ##
## I WILL ALSO BE VISUALIZING AND PRE-PROCESSING MY IMAGE DATA HERE ##

import torch
import torch.nn 
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image


## Lets prepare our data:

class Data(Dataset):
    def __init__(self,root_dir,transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        real_dir = os.path.join(root_dir,'real')
        fake_dir = os.path.join(root_dir,'fake')
        
        for img_name in os.listdir(real_dir):
            self.images.append(os.path.join(real_dir,img_name))
            self.labels.append(1) # 1 for real
            
        for img_name in os.listdir(fake_dir):
            self.images.append(os.path.join(fake_dir,img_name))
            self.labels.append(0)
          
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,ind):
        img_path = self.images[ind]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[ind]
        
        if self.transform:
            image = self.transform(image)
            
        return image,label

   


