import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
from torch.utils import data
from dgl import DGLGraph
import helper
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import EdgeConv



class graphData(data.Dataset):
  def __init__(self,direc):
    with open(direc, 'rb') as f:
      self.data = pickle.load(f)
    self.graph = helper.makeGraphOnly()

  def __len__(self):
    return len(self.data.keys())

  def __getitem__(self, index):
  # Select sample
    return self.graph,helper.makeFeature(self.data[index])



class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.mean(msg='m', out='h')
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = EdgeConv(2,16)
        self.layer2 = GraphConv(16, 8)
        self.layer4 = GraphConv(8, 2)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = F.relu(self.layer2(g, features))
        x = self.layer4(g, x)
        return x




if __name__ == "__main__":
  # direc = "../data/triangles.pickle"
  # with open(direc, 'rb') as f:
  #     data = pickle.load(f)


  # graph = trajGraph(data[0])
  # print(graph.adj_matrix)
  # print(graph.property_matrix)
  # print(graph.neighbours)
  pass

