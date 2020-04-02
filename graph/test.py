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
net = Net()
data = graphData(direc)
net.load_state_dict(torch.load("model.pth"))
edges,features = data[190000]
g = dgl.DGLGraph(edges)
logits = net(g, features).tolist()
features = features.tolist()

plt.figure(1)
nx.draw(g.to_networkx(),pos=features, with_labels=True)

plt.figure(2)
nx.draw(g.to_networkx(),pos=logits, with_labels=True)

plt.show()
