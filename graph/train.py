from helper import *
import pickle
import torch
import dgl
import matplotlib.pyplot as plt
import networkx as nx
from classes import *
from torch.utils.data import DataLoader
import pickle
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F
import numpy as np



direc = "../data/triangles.pickle"
# with open(direc, 'rb') as f:
#   data = pickle.load(f)
# direc2 = "../data/graphs.pickle"

epochs = 200
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
writerb = SummaryWriter('runs/slp')

data =  graphData(direc)
# train_loader = DataLoader(data,batch_size=None)
# test_loader = DataLoader(data,batch_size=None)
net = Net()

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
dur = []
for epoch in range(epochs):
  if epoch >=3:
      t0 = time.time()
  iterer = np.random.permutation(1000)
  for i in iterer:
    # print(i)
    g,features = data[i]
    net.train()
    optimizer.zero_grad()
    logits = net(g, features)
    loss = F.mse_loss(logits,features)    
    
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
  torch.save(net.state_dict(), "model.pth")

# print(graph.node_attr_schemes())
# nx.draw(graph.to_networkx(), with_labels=True)
# plt.show()
