## MODEL ARCHITECTURE GOES HERE ##

from vis import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image


class Deepfake(nn.Module):
    def __init__(self):
        super(Deepfake, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*28*28, 512),  # Corrected line
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Assuming binary classification (real/fake)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
