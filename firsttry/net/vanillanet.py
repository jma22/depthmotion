import os
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import pickle
from torch.utils import data
from torch_scatter import scatter


class messNet(nn.Module):
  def __init__(self):
    super(messNet, self).__init__()
    self.fc1 = nn.Linear(6, 1)
    self.fc2 = nn.Linear(6, 1)
    # self.fc3 = nn.Linear(6, 6)
    self.fc4 = nn.Linear(6, 1)

  def forward(self, edges,coor):
    # print(x[1][0])
    new = []
    for row in coor[0]:
      a = self.fc1(row)
      b = self.fc2(row)
      # row = self.fc3(row)
      c = self.fc4(row)
      temp = torch.stack([a,b,c])
      new.append(temp[0])
    vals = torch.stack(new)
    print(vals.shape)
    out = scatter(vals.t(), edges[0][:,-1], dim=1, reduce="mean")
    # print(out.shape)
    return torch.unsqueeze(out.t(),0)
