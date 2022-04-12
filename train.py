import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
import torch.nn.functional as F
import torchvision.transforms as tt
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from Datasets import FER2013
import argparse
from resnet9 import ResNet18
import Utils as utils
from torch.autograd import Variable

# Check device, use cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# various data augmentation techniques. Online respos have separate methods for these typically
#in a script named transforms
train_tfms = tt.Compose([tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.Resize((150,150)),
                         tt.RandomCrop(150, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.RandomRotation(10),
                         tt.ToTensor()])
                        # tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.Resize((150,150)),tt.ToTensor()])#, tt.Normalize(*stats)])

# Datasplit occurs in the Dataset class, custom made for FER2013 using Pytorch Abstract class
train_ds = FER2013(split='Training', transform=train_tfms)
valid_ds = FER2013(split='PublicTest', transform=valid_tfms)
test_ds = FER2013(split='Private_test', transform=valid_tfms)

# labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
# [1 .. 0] = 1, [0 1 .. 0] = 2, [0 0 1 .. 0] = 3, etc.

# Dataloader "Partitions" the data into batches, can use shuffle in order to randomize what elements are in which
# batch (i.e. the space is not simply equally partitioned)
batch_size = 100
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, 2*batch_size, num_workers=3, pin_memory=True)

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:32], nrow=8).permute(1, 2, 0))
        break

if __name__ == '__main__':
    show_batch(valid_dl)
    plt.show()