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
# from dgl.nn.pytorch import customConv




class gateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GraphConv(4,64)
        self.layer2 = GraphConv(32,64)
        self.layer3 = GraphConv(32,64)
        self.layer4 = GraphConv(32,64)
        self.layer5 = GraphConv(32,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(F.glu(self.layer1(g, features)))
        x = F.leaky_relu(F.glu(self.layer2(g, x)))
        x = F.leaky_relu(F.glu(self.layer3(g, x)))
        x = F.leaky_relu(F.glu(self.layer4(g, x)))
        x = self.layer5(g, x)
        
     
        return x

class deepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GraphConv(4,64)
        self.layer2 = GraphConv(64,64)
        self.layer3 = GraphConv(64,64)
        self.layer4 = GraphConv(64,64)
        self.layer5 = GraphConv(64,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.leaky_relu(self.layer4(g, x))
        x = self.layer5(g, x)
        
     
        return x


class deeperNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GraphConv(4,64)
        self.layer2 = GraphConv(64,64)
        self.layer3 = GraphConv(64,64)
        self.layer4 = GraphConv(64,64)
        self.layer5 = GraphConv(64,64)
        self.layer6 = GraphConv(64,64)
        self.layer7 = GraphConv(64,64)
        self.layer8 = GraphConv(64,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.leaky_relu(self.layer4(g, x))
        x = F.leaky_relu(self.layer5(g, x))
        x = F.leaky_relu(self.layer6(g, x))
        x = F.leaky_relu(self.layer7(g, x))
        x = self.layer8(g, x)
        
     
        return x
    
class shallowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GraphConv(4,64)
        self.layer2 = GraphConv(64,3)
 
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        
        x = self.layer2(g, x)
        
     
        return x
    

class deepedgeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = customConv(3,64)
        self.layer2 = customConv(64,64)
        self.layer3 = customConv(64,64)
        self.layer4 = customConv(64,64)
        self.layer5 = customConv(64,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.leaky_relu(self.layer4(g, x))
        x = self.layer5(g, x)
        
     
        return x


    
class gatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GATConv(3,64,1)
        self.layer2 = GATConv(64,64,1)
        self.layer3 = GATConv(64,3,1)
        
        
         


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = self.layer3(g, x)
        
        

        return x[:,0,:]

    
# class customNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = customConv(3,32)
    
#         self.layer4 = customConv(32,3)
        
        
        
#         # self.dropout = nn.Dropout(p=0.6)
      


#     def forward(self, g, features):
#         # dropped = self.dropout(features)
#         x = self.layer1(g, features)
#         x = self.layer4(g, x)
        
        

#         return x



    
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