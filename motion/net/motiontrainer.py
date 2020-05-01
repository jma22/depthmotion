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
  sg = DGLGraph(spatial_g)                       ##K NEAREST##
  tg = DGLGraph(temporal_g)   
  model.eval()
  with torch.no_grad():
    wholeloss = 0
    reconsloss = 0
    for i in np.random.permutation(list(range(5500,6000))):
        #normal
#       g= data[i]["edge"]  #.to(device=device, non_blocking=True)
#       features= data[i]["coor"]  #.to(device=device, non_blocking=True)
#       features = torch.Tensor(features).to(device=device, non_blocking=True)
#reconstruction 
        features = data[i]["coor"] ##lsit of 20
        seen = data[i]["visible"] ##lsit of 20
        newfeatures = []
        for framecoor in features:
            newfeatures.append(np.hstack((framecoor,np.zeros((588,1)))))
        for framenum in range(len(newfeatures)):                        
            for idx in range(newfeatures[framenum].shape[0]):
               if idx not in seen[framenum]:
                   newfeatures[framenum][idx] = np.array([0,0,0,1])
    
#         y = features[target:target+1].copy()
        features = torch.Tensor(np.vstack(features)).to(device=device, non_blocking=True)
        newfeatures = torch.Tensor(np.vstack(newfeatures)).to(device=device, non_blocking=True)
#         y = torch.Tensor(y).to(device=device, non_blocking=True)
        logits = model(sg,tg, newfeatures)
        wholeloss+= F.l1_loss(logits,features) 

  return wholeloss.item()/500
  #######
data_path = '../data/move_cube_data.pickle'


with open(data_path,"rb") as f:
  data = pickle.load(f)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')




train_losses = []
epochs = 300
dur = []



k= 8                         ##K NEAREST##
spatial_g= knn_graph_edges(k,20)    
temporal_g= temporal_edges(20) ##K NEAREST##
writerb = SummaryWriter('runs/stnet'.format(k))    ###CHANGE  THIS#####
net= stNet().to(device)                   ###CHANGE  THIS#####
optimizer =torch.optim.SGD(net.parameters(), lr=0.01)
net.train()

for epoch in range(epochs):
  if epoch >=3:
      t0 = time.time()
  # iterer = np.random.permutation(50000)
  if epoch <3:
    iterer = np.random.permutation(100)
  else:
    iterer = np.random.permutation(6000)
    ### training loop ###
  for i in iterer:
    """editing data"""
    features = data[i]["coor"] ##lsit of 20
    seen = data[i]["visible"] ##lsit of 20
    newfeatures = []
    for framecoor in features:
        newfeatures.append(np.hstack((framecoor,np.zeros((588,1)))))
    for framenum in range(len(newfeatures)):                        
        for idx in range(newfeatures[framenum].shape[0]):
           if idx not in seen[framenum]:
               newfeatures[framenum][idx] = np.array([0,0,0,1])
    
    
    sg = DGLGraph(spatial_g)
    tg = DGLGraph(temporal_g)
    features = torch.Tensor(np.vstack(features)).to(device=device, non_blocking=True)
    newfeatures = torch.Tensor(np.vstack(newfeatures)).to(device=device, non_blocking=True)
#     y = torch.Tensor(y).to(device=device, non_blocking=True)
    """end"""


    net.train()
    optimizer.zero_grad()
    output = net(sg,tg, newfeatures)
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
  torch.save(net.state_dict(), "models/st8.pth")                 ###CHANGE  THIS#####


