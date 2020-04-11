import os
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import dgl
import dgl.function as fn
from torch.utils.data import DataLoader
from torch.utils import data
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import EdgeConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import customConv

class incustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = customConv(6,
                                32)
    
        self.layer4 = customConv(32,3)
        
        
        
        # self.dropout = nn.Dropout(p=0.6)
      


    def forward(self, g, features):
        # dropped = self.dropout(features)
        x = self.layer1(g, features)
        x = self.layer4(g, x)
        
        

        return x

    
class customNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = customConv(3,32)
    
        self.layer4 = customConv(32,3)
        
        
        
        # self.dropout = nn.Dropout(p=0.6)
      


    def forward(self, g, features):
        # dropped = self.dropout(features)
        x = self.layer1(g, features)
        x = self.layer4(g, x)
        
        

        return x

    
class deepcustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = customConv(3,16)
        self.layer2 = customConv(16,16)
        self.layer3 = customConv(16,16)
        self.layer4 = customConv(16,3)
        
        
        
        # self.dropout = nn.Dropout(p=0.6)
      


    def forward(self, g, features):
        # dropped = self.dropout(features)
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        x = self.layer3(g, x)
        x = self.layer4(g, x)
        
        

        return x


class customDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = customConv(3,16)
        self.layer2 = customConv(16,16)
        self.layer3 = customConv(16,1)
        self.fc1 = nn.Linear(1764//3,1764//3)
        self.fc2 = nn.Linear(1764//3,1764//3)
        self.fc3 = nn.Linear(1764//3,1764)

        
        # self.dropout = nn.Dropout(p=0.6)
      


    def forward(self, g, features):
        # dropped = self.dropout(features)
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        x = self.layer3(g, x)
        x = x.view(1,-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1,3)
        

        return x
    
    
class try10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = EdgeConv(3,3)
        self.layer2 = EdgeConv(3,1)
        self.fc1 = nn.Linear(1764//3,10)
        self.fc2 = nn.Linear(10,128)
        self.fc3 = nn.Linear(128,256)
        self.fc4 = nn.Linear(256,512)
        self.fc5= nn.Linear(512,1024)
        self.fc6 = nn.Linear(1024,1764)
        
        # self.dropout = nn.Dropout(p=0.6)
      


    def forward(self, g, features):
        # dropped = self.dropout(features)
        x = self.layer1(g, features)
        x = self.layer2(g, x)
        x = x.view(1,-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = self.fc6(x)
        x = x.view(-1,3)
        

        return x
    

    
class narrowsagemNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = SAGEConv(3,32,aggregator_type='mean')
        self.layer3 = SAGEConv(32, 3,aggregator_type='mean')
   
        # self.dropout = nn.Dropout(p=0.6)
      


    def forward(self, g, features):
        # dropped = self.dropout(features)
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer3(g, x))

        return x


if __name__ == "__main__":
    pass
    #scatter eg
    # index = torch.tensor([0, 1, 0, 1, 0, 1])
    # val =  torch.tensor([[0, 1, 0, 1, 0, 1],[0, 1, 0, 1, 0, 1]])
    # print(val.shape)
    # out = scatter(val, index, dim=1, reduce="sum")
    # print(out)
    # data_path = '../datagen/data/cube_data.pickle'
    # labels_path = '../datagen/data/cube_labels.pickle'
    # graph_path = '../datagen/data/cube_graph.pickle'
    # train = GraphData(range(0,20000),data_path,labels_path,graph_path)
    # print(train[2])