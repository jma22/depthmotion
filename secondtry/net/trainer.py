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
######
def evaluate(model, data):
  extra = torch.tensor(np.ones((588,3)))
  model.eval()
  with torch.no_grad():
    loss = 0
    for i in np.random.permutation(list(range(98000,100000))):
      g= data[i]["edge"]  #.to(device=device, non_blocking=True)
      features= data[i]["coor"]  #.to(device=device, non_blocking=True)
      features = torch.Tensor(features).to(device=device, non_blocking=True)
      g = DGLGraph(g)
      logits = model(g, features)
      loss+= F.mse_loss(logits,features) 

  return loss.item()/2000
  #######
data_path = '../data/cube_data.pickle'
# model_path = '../models/model.pth'
with open(data_path,"rb") as f:
  data = pickle.load(f)


writerb = SummaryWriter('runs/incustomnet')    ###CHANGE  THIS#####

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

net= incustomNet().to(device)                   ###CHANGE  THIS#####

optimizer =torch.optim.SGD(net.parameters(), lr=0.01)
net.train()
train_losses = []
epochs = 300
dur = []
extra = torch.tensor(np.ones((588,3)))
for epoch in range(epochs):
  if epoch >=3:
      t0 = time.time()
  # iterer = np.random.permutation(50000)
  if epoch <3:
    iterer = np.random.permutation(100)
  else:
    iterer = np.random.permutation(50000)
  for i in iterer:
    g= data[i]["edge"] 
    features= data[i]["coor"] 
    

    g = DGLGraph(g)
    features = torch.Tensor(features).to(device=device, non_blocking=True)

    net.train()
    optimizer.zero_grad()
    output = net(g, features)
    # loss = F.mse_loss(output,features)
    loss = F.mse_loss(output,features) + F.l1_loss(output,features)
    loss.backward()
    optimizer.step()

  if epoch >=3:
    dur.append(time.time() - t0)
  acc = evaluate(net, data)
  # acc=4
  writerb.add_scalar('training loss',acc,epoch)
  # writerb.add_scalar('training acc',acc,epoch)
  print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
          epoch, loss.item(), acc, np.mean(dur)))
  torch.save(net.state_dict(), "models/incustomnet.pth")                 ###CHANGE  THIS#####

# print(graph.node_attr_schemes())
# nx.draw(graph.to_networkx(), with_labels=True)
# plt.show()

