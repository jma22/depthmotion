import classes
import os
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pickle
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
def plot4(train_loader):
  with torch.no_grad():
    dataiter = iter(train_loader)
    fig, axs = plt.subplots(2, 4)
    for x in range(4):
      data, labels = dataiter.next()
      output = model(data)
      axs[0, x].plot(data[0])
      axs[1, x].plot(output[0], 'tab:orange')
      
    axs[0,0].set(ylabel='data')
    axs[0,0].set(ylabel='modelout')
    plt.show()

def plotonepair(fig,axs,units,train_loader):
  with torch.no_grad():
    dataiter = iter(train_loader)
    data, labels = dataiter.next()
    output = model(data)
    axs[0, 5-units].plot(data[0])
    axs[1, 5-units].plot(output[0], 'tab:orange')
    axs[0,0].set(ylabel='data')
    axs[0,1].set(ylabel='modelout')
    axs[1, 5-units].set(xlabel='{}units'.format(units))


# default `log_dir` is "runs" - we'll be more specific here

epochs = 2
direc = "data/series.pickle"
use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')
writerb = SummaryWriter('runs/cnn')



data =  classes.seriesData(direc)
train_loader = DataLoader(data,batch_size=1,shuffle=False)
plot_loader = DataLoader(data,batch_size=None,shuffle=False)
fig, axs = plt.subplots(2, 5)
compressed = 5

for hidden in range(7,0,-2):
  writer = SummaryWriter('runs/mlp{}'.format(hidden))
  model = classes.conv(kernel = hidden).to(device)
  model = model.double()  
  optimiser = torch.optim.SGD(model.parameters(),lr = 0.05)
  for epoch in range(epochs):
    for i, (data,target) in enumerate(train_loader):
      data = data.to(device=device, non_blocking=True)
      target = target.to(device=device, non_blocking=True)
      # print(data)
      #forward
      output = model(data)
      loss = F.mse_loss(output, target)
      
      ## backward
      optimiser.zero_grad()
      loss.backward()
      optimiser.step()
      if i%200==0:
        print(loss.item())
        writer.add_scalar('training loss',loss.item(),i +epoch*20000)
      if i ==20000:
        break
  # plotonepair(fig,axs,hidden,plot_loader)
  writerb.add_scalar('loss vs units',loss.item(),hidden)
# plt.show()
# plot4(plot_loader)



  
  



# run tensorboard --logdir=runs
