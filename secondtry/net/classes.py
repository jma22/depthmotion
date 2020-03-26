import os
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import pickle
from torch.utils import data
from torch_geometric.nn import GCNConv
from torch_scatter import scatter


class data2(data.Dataset):
    def __init__(self, list_IDs, data_path,labels_path):
        with open(data_path,"rb") as f:
            self.data = pickle.load(f)
        with open(labels_path,"rb") as f:
            self.truth = pickle.load(f)

        self.IDs = set(list_IDs)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
    # Select sample
        # if index in self.IDs:
        edges = torch.LongTensor(self.data[index]['edge'])
        coor = torch.FloatTensor(self.data[index]['coor'])
        # cam = torch.FloatTensor(self.data[index]['cam'])

        label = torch.FloatTensor(self.truth[index])
        # print(self.graph.T)
        # print(x)
        return edges,coor,label


class GraphData(data.Dataset):
    def __init__(self, list_IDs, data_path,labels_path,graph_path):
        with open(data_path,"rb") as f:
            self.data = pickle.load(f)
        with open(labels_path,"rb") as f:
            self.labels = pickle.load(f)
        with open(graph_path,"rb") as f:
            self.graph = pickle.load(f)
        self.IDs = set(list_IDs)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
    # Select sample
        # if index in self.IDs:
        x = torch.FloatTensor(self.data[index]['x'],)
        # print(x.shape)
        # print(np.max(self.graph))
        # edge_index = torch.FloatTensor(self.data[index]['edge_index'])
        label = torch.FloatTensor(self.labels[index])
        graph = torch.LongTensor(self.graph.T)
        # print(self.graph.T)
        # print(x)
        return [x,graph],label

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