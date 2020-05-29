import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import time
import numpy as np
import helpers



## Set device
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

## Init Parameters
name = "stategcn4"
dur = []
k = 16
frames = 90
lr=0.001
epochs = 300
writerb = SummaryWriter('runs/{}'.format(name))

## Make model
net= cust4Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

##premake graphs
spatial_graph = knn_graph(k,frames)  ##TODO: Make
temporal_graph = temp_graph(frames)  ##TODO: Make

##Epoch loop
for epoch in range(epochs):
    #timer
    if epoch >=3:
        t0 = time.time()
    # shuffle
    iterater = make_shuffled_idx(train_start,train_end)  ##TODO: Make
    ## Training - n times
    for idx in iterater:
        optimizer.zero_grad()
        train_loss = get_loss(idx,net,k,frames,device,train = True,spatial_graph,temporal_graph) 
        train_loss.backward()
        optimizer.step()

    # save dur
    if epoch >=3:
        dur.append(time.time() - t0)
    ## evaluate
    test_loss = evaluate(net, test_start,test_end) ##return loss TODO:make
    # log and save model
    writerb.add_scalar('test_loss',test_loss,epoch)
    print("Epoch {:05d} | Train_loss {:.4f} | Test_loss {:.4f}  | Time(s) {:.4f}".format(epoch, train_loss.item(), test_loss, np.mean(dur)))
    torch.save(net.state_dict(), "models/{}.pth".format(name))




