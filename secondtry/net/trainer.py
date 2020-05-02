import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from classes import *
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dgl import DGLGraph
import time
from knn import *

######
def evaluate(model, data):
  g=  knn_graph_edges(k)                        ##K NEAREST##
  model.eval()
  with torch.no_grad():
    wholeloss = 0
    reconsloss = 0
    for i in np.random.permutation(list(range(28000,30000))):
        #normal
#       g= data[i]["edge"]  #.to(device=device, non_blocking=True)
#       features= data[i]["coor"]  #.to(device=device, non_blocking=True)
#       features = torch.Tensor(features).to(device=device, non_blocking=True)
#reconstruction 
        features = data[i]["coor"]/4
        newfeatures= np.hstack((data[i]["coor"],np.zeros((features.shape[0],1))))
        for idx,col in enumerate(newfeatures):
            if idx not in data[i]['visible']:
                newfeatures[idx] = np.array([0,0,0,1])
#         y = features[target:target+1].copy()
        features = torch.Tensor(features).to(device=device, non_blocking=True)
        newfeatures = torch.Tensor(newfeatures).to(device=device, non_blocking=True)
#         y = torch.Tensor(y).to(device=device, non_blocking=True)
        
        g = DGLGraph(g)
        logits = model(g, newfeatures)
        wholeloss+= F.l1_loss(logits,features) 

  return wholeloss.item()/2000
  #######
data_path = '../data/monkey_data.pickle'


with open(data_path,"rb") as f:
  data = pickle.load(f)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')




train_losses = []
epochs = 300
dur = []


# for l in range(8,40,4):
k= 16                           ##K NEAREST##
g= knn_graph_edges(k)                        ##K NEAREST##
writerb = SummaryWriter('runs/deep{}net'.format(k))    ###CHANGE  THIS#####
net= deepNet().to(device)                   ###CHANGE  THIS#####
optimizer =torch.optim.SGD(net.parameters(), lr=0.01)
net.train()

for epoch in range(epochs):
  if epoch >=3:
      t0 = time.time()
  # iterer = np.random.permutation(50000)
  if epoch <3:
    iterer = np.random.permutation(100)
  else:
    iterer = np.random.permutation(28000)
  for i in iterer:
#     g= data[i]["edge"]                      ##K NEAREST##
    features = data[i]["coor"]/4
    newfeatures= np.hstack((data[i]["coor"],np.zeros((features.shape[0],1))))
    for idx in range(newfeatures.shape[0]):
       if idx not in data[i]['visible']:
           newfeatures[idx] = np.array([0,0,0,1])
#     y = features[target:target+1].copy()


    g = DGLGraph(g)
    features = torch.Tensor(features).to(device=device, non_blocking=True)
    newfeatures = torch.Tensor(newfeatures).to(device=device, non_blocking=True)
#     y = torch.Tensor(y).to(device=device, non_blocking=True)


    net.train()
    optimizer.zero_grad()
    output = net(g, newfeatures)
    # loss = F.mse_loss(output,features)
#     loss = F.mse_loss(output,features) + F.l1_loss(output,features)

    loss = F.mse_loss(output,features) + F.l1_loss(output,features)
    loss.backward()
    optimizer.step()

  if epoch >=3:
    dur.append(time.time() - t0)
  acc = evaluate(net, data)
  # acc=4
  writerb.add_scalar('whole loss',acc,epoch)

  print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
          epoch, loss.item(), acc, np.mean(dur)))
  torch.save(net.state_dict(), "models/deep{}.pth".format(k))                 ###CHANGE  THIS#####


