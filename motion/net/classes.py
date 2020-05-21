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


from dgl.nn.pytorch import CustomConv

class lastgcn(nn.Module):
    def __init__(self,cap=30):
        super().__init__()
        self.layer1 = CustomConv(4,64)
        self.layer2 = CustomConv(64,64)
        self.layer3 = CustomConv(64,1)
        self.state_handle = nn.GRUCell(588, 64)
        self.end_determine = nn.Linear(64,1)
        self.layer5 = CustomConv(64,3)
        
        self.cap = cap
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        counter = 0
        self.state = torch.zeros(1,64).to(self.device)
        x = F.leaky_relu(self.layer1(g, features))
        inputt = F.leaky_relu(self.layer3(g, x)).t()
        self.state = self.state_handle(inputt,self.state)
        preprob = self.end_determine(self.state)
        prob = torch.sigmoid(preprob)
        probm = torch.unsqueeze(preprob, 0)
        fprob = prob
        final = torch.unsqueeze(x, 0)
        while torch.sum(torch.sigmoid(probm)) < 1-0.001:
            if counter >=self.cap:
                break
            x = F.leaky_relu(self.layer2(g, x))
            preprob = self.end_determine(self.state_handle(F.leaky_relu(self.layer3(g, x)).t(),self.state))
            prob = torch.sigmoid(preprob)
            fprob = prob
            final = torch.cat((final,torch.unsqueeze(x, 0)))
            probm = torch.cat((probm,torch.unsqueeze(preprob, 0)))
            counter+=1

        if torch.sum(torch.sigmoid(probm)) > 1:
            probm[-1,:,:] = torch.log((1-torch.sum(torch.sigmoid(probm[:-1,:,:])))/torch.sum(torch.sigmoid(probm[:-1,:,:])))
        probma = torch.sigmoid(probm)
        last = final[-1,:,:]  
        x = self.layer5(g, last)

        return x,counter,fprob
    
    
class stategcn(nn.Module):
    def __init__(self,cap=30):
        super().__init__()
        self.layer1 = CustomConv(4,64)
        self.layer2 = CustomConv(64,64)
        self.layer3 = CustomConv(64,1)
        self.state_handle = nn.GRUCell(588, 64)
        self.end_determine = nn.Linear(64,1)
        self.layer5 = CustomConv(64,3)
        
        self.cap = cap
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        counter = 0
        self.state = torch.zeros(1,64).to(self.device)
        x = F.leaky_relu(self.layer1(g, features))
        inputt = F.leaky_relu(self.layer3(g, x)).t()
        self.state = self.state_handle(inputt,self.state)
        preprob = self.end_determine(self.state)
        prob = torch.sigmoid(preprob)
        probm = torch.unsqueeze(preprob, 0)
        fprob = prob
        final = torch.unsqueeze(x, 0)
        while torch.sum(torch.sigmoid(probm)) < 1-0.001:
            if counter >=self.cap:
                break
            x = F.leaky_relu(self.layer2(g, x))
            preprob = self.end_determine(self.state_handle(F.leaky_relu(self.layer3(g, x)).t(),self.state))
            prob = torch.sigmoid(preprob)
            fprob = prob
            final = torch.cat((final,torch.unsqueeze(x, 0)))
            probm = torch.cat((probm,torch.unsqueeze(preprob, 0)))
            counter+=1

        if torch.sum(torch.sigmoid(probm)) > 1:
            probm[-1,:,:] = torch.log((1-torch.sum(torch.sigmoid(probm[:-1,:,:])))/torch.sum(torch.sigmoid(probm[:-1,:,:])))
        probma = torch.sigmoid(probm)
        last = final*probma
        last = torch.sum(last,0)    
        x = self.layer5(g, last)

        return x,counter,fprob