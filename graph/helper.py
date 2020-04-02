import dgl
import torch
import torch.nn.functional as F
import numpy as np

def makeDGL(array):
  nodes = []
  for frame in array:
    for point in frame:
      nodes.append(point)
  edges = []
  for i in range(10):
    edges.append((0+i*3,1+i*3))
    edges.append((1+i*3,2+i*3))
    edges.append((0+i*3,2+i*3))
    edges.append((1+i*3,0+i*3))
    edges.append((2+i*3,1+i*3))
    edges.append((2+i*3,0+i*3))
    edges.append((0+i*3,0+i*3))
    edges.append((1+i*3,1+i*3))
    edges.append((2+i*3,2+i*3))
    if i !=9:
      edges.append((0+(i)*3,0+(i+1)*3))
      edges.append((1+(i)*3,1+(i+1)*3))
      edges.append((2+(i)*3,2+(i+1)*3))
    if i !=0:
      edges.append((0+(i-1)*3,0+(i)*3))
      edges.append((1+(i-1)*3,1+(i)*3))
      edges.append((2+(i-1)*3,2+(i)*3))
  graph = dgl.DGLGraph(edges)
  graph.ndata['pos'] = torch.tensor(nodes)
  return graph

def evaluate(model, data):
  model.eval()
  with torch.no_grad():
    loss = 0
    for i in np.random.permutation(list(range(195000,200000))):
      g,features = data[i]
      logits = model(g, features)
      loss+= F.mse_loss(logits,features) 

  return loss.item()/5000

def makeGraphOnly():

  edges = []
  for i in range(10):
    edges.append((0+i*3,1+i*3))
    edges.append((1+i*3,2+i*3))
    edges.append((0+i*3,2+i*3))
    if i !=9:
      edges.append((0+(i)*3,0+(i+1)*3))
      edges.append((1+(i)*3,1+(i+1)*3))
      edges.append((2+(i)*3,2+(i+1)*3))
  graph = dgl.DGLGraph(edges)
  return graph
def makeFeature(array):
  nodes = []
  for frame in array:
    for point in frame:
      nodes.append(point)
  lol = torch.FloatTensor(nodes)
  return lol


