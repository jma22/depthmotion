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
    self.fc1 = nn.Linear(3, 3)
    self.fc2 = nn.Linear(3, 3)
    # self.fc3 = nn.Linear(6, 6)
    self.fc4 = nn.Linear(3, 3)

  def forward(self, edges,coor):
    # print(x[1][0])
    new = []
    for row in coor[0]:
      row = self.fc1(row[0:3])
      row = self.fc2(row)
      # row = self.fc3(row)
      row = self.fc4(row)
      new.append(row)
    vals = torch.stack(new)
    print(vals.shape)
    out = scatter(vals.t(), edges[0][:,-1], dim=1, reduce="mean")
    # print(out.shape)
    return torch.unsqueeze(out.t(),0)
