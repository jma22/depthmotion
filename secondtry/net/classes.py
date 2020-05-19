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

# class rgcn(nn.Module):
#     def __init__(self,cap=30):
#         super().__init__()
#         self.layer1 = CustomConv(4,64)
#         self.layer2 = CustomConv(64,64)
#         self.layer3 = CustomConv(64,1)
#         self.end_determine = nn.Linear(588,1)
#         self.layer5 = CustomConv(64,3)
        
#         self.cap = cap
        
        
        
# #         self.dropout = nn.Dropout(p=0.5)
      


#     def forward(self, g, features):
#         counter = 0
#         x = F.leaky_relu(self.layer1(g, features))
#         prob = F.leaky_relu(self.layer3(g, x)).t()
#         prob = self.end_determine(prob)
#         torch.sigmoid(prob)
#         probm = torch.unsqueeze(prob, 0)
#         cum_prob =prob
#         fprob = prob
#         final = torch.unsqueeze(x, 0)
#         while cum_prob < 1-0.001:
#             if counter >=self.cap:
#                 probm[-1,:,:] = 1-cum_prob
#                 break
#             x = F.leaky_relu(self.layer2(g, x))
#             prob = self.end_determine(F.leaky_relu(self.layer3(g, x)).t())
#             torch.sigmoid(prob)
#             fprob = prob
#             final = torch.cat((final,torch.unsqueeze(x, 0)))
#             probm = torch.cat((probm,torch.unsqueeze(prob, 0)))
#             cum_prob +=prob
#             counter+=1
#         probm[-1,:,:] = 1-cum_prob
#         last = final*probm
#         last = torch.sum(last,0)      
#         x = self.layer5(g, last)

#         return x,counter,fprob
class deepcustNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = CustomConv(4,64)
        self.layer2 = CustomConv(64,64)
        self.layer5 = CustomConv(64,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = self.layer5(g, x)
        
     
        return x
    
class cust4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = CustomConv(4,64)
        self.layer2 = CustomConv(64,64)
        self.layer5 = CustomConv(64,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))

        x = self.layer5(g, x)
        
     
        return x

class repeatdeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GraphConv(4,64)
        self.layer2 = GraphConv(64,64)
        self.layer5 = GraphConv(64,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer2(g, x))
        x = self.layer5(g, x)
        
     
        return x

class customNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = CustomConv(4,64)
        self.layer2 = CustomConv(64,64)
        self.layer3 = CustomConv(64,64)
        self.layer4 = CustomConv(64,64)
        self.layer5 = CustomConv(64,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.leaky_relu(self.layer4(g, x))
        x = self.layer5(g, x)
        
     
        return x
    

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
    
    
class deepwideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = GraphConv(4,128)
        self.layer2 = GraphConv(128,128)
        self.layer3 = GraphConv(128,128)
        self.layer4 = GraphConv(128,128)
        self.layer5 = GraphConv(128,3)
        
        
        
#         self.dropout = nn.Dropout(p=0.5)
      


    def forward(self, g, features):
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.leaky_relu(self.layer4(g, x))
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



    
class sageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = SAGEConv(4,32,aggregator_type='mean')
        self.layer2 = SAGEConv(32,32,aggregator_type='mean')
        self.layer3 = SAGEConv(32,32,aggregator_type='mean')
        self.layer4 = SAGEConv(32,32,aggregator_type='mean')
        
        self.layer5 = SAGEConv(32, 3,aggregator_type='mean')
   
        # self.dropout = nn.Dropout(p=0.6)
      


    def forward(self, g, features):
        # dropped = self.dropout(features)
        x = F.leaky_relu(self.layer1(g, features))
        x = F.leaky_relu(self.layer2(g, x))
        x = F.leaky_relu(self.layer3(g, x))
        x = F.leaky_relu(self.layer4(g, x))
        x = self.layer5(g, x)

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