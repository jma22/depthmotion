import os
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
from torch.utils import data


class seriesData(data.Dataset):
  def __init__(self,direc):
    with open(direc, 'rb') as f:
      self.data = pickle.load(f)

  def __len__(self):
    return len(self.data.keys())

  def __getitem__(self, index):
  # Select sample
    return torch.tensor(self.data[index]),torch.tensor(self.data[index])
class slp(nn.Module):
  def __init__(self,**kwargs):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(kwargs["input_size"],kwargs["hidden"]),
      nn.ReLU(True))
    self.decoder = nn.Sequential(             
      nn.Linear(kwargs["hidden"],kwargs["input_size"]),
      nn.ReLU(True),
      )
  def forward(self,x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class mlp(nn.Module):
  def __init__(self,**kwargs):
    super().__init__()
    self.encoder = nn.Sequential(
      nn.Linear(kwargs["input_size"],kwargs["hidden"]),
      nn.ReLU(True),
      nn.Linear(kwargs["hidden"],kwargs["compressed"]),
      nn.ReLU(True))
    self.decoder = nn.Sequential( 
      nn.Linear(kwargs["compressed"],kwargs["hidden"]),
      nn.ReLU(True),            
      nn.Linear(kwargs["hidden"],kwargs["input_size"]),
      nn.ReLU(True)
      )
  def forward(self,x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class conv(nn.Module):
  def __init__(self,**kwargs):
    super().__init__()
    self.conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kwargs["kernel"], stride=1, padding=kwargs["kernel"]//2)
    self.encoder = nn.Sequential(
      
      torch.nn.MaxPool1d(kernel_size=2),
      nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kwargs["kernel"], stride=1, padding=kwargs["kernel"]//2),
      torch.nn.MaxPool1d(kernel_size=5),
      nn.Flatten(),
      nn.Linear(32,5)
    )
    self.fc1 = nn.Linear(5,32)
    self.upsample = nn.Upsample(size=32*5)
    self.convinv1 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=kwargs["kernel"], stride=1, padding=kwargs["kernel"]//2)
    self.up2 = nn.Upsample(scale_factor = 2)
    self.convinv2 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=kwargs["kernel"], stride=1, padding=kwargs["kernel"]//2)


  def forward(self,x):
    x=self.conv(x)
    x = self.encoder(x)
    
    x= self.fc1(x)
    x = self.upsample(x.unsqueeze(0)).view(1,32,5)
    
    x= self.convinv1(x)
    x = self.up2(x)
    x= self.convinv2(x)
    return x


# asd = conv(kernel=9)
# print(asd.forward(torch.ones(1,1,10)))