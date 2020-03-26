from pytorch_lightning import Trainer
from vanillanet import messNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from classes import data2
from torch.utils.data import DataLoader

data_path = '../datagen/data2/cube_data_short.pickle'
truth_path = '../datagen/data2/cube_truth_short.pickle'

# data_path = '../datagen/data2/cube_data_short.pickle'
# truth_path = '../datagen/data2/cube_truth_short.pickle'
# with open(data_path,"rb") as f:
#   data = pickle.load(f)
# with open(truth_path,"rb") as f:
#   truth = pickle.load(f)
train_data = data2(list(range(0,1000)),data_path,truth_path)
# print(train_data[0])
# test_data = datasets.MNIST(data_path, train=False, transform=transforms.Compose([
#                              transforms.ToTensor(),
#                              transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(train_data, batch_size=1,
                          shuffle=True, num_workers=4, pin_memory=True)
# test_loader = DataLoader(test_data, batch_size=args.batch_size,
#                          num_workers=4, pin_memory=True)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
model = messNet().to(device)
optimiser =torch.optim.SGD(model.parameters(), lr=0.01)
model.train()
train_losses = []
# output = model(data)
for epoch in range(10):
  for i, (edges,coor,label) in enumerate(train_loader):
    print("num:" + str(i))
    print("epoch:" + str(epoch))

    edges = edges.to(device=device, non_blocking=True)
    coor = coor.to(device=device, non_blocking=True)
    label = label.to(device=device, non_blocking=True)
    optimiser.zero_grad()
    output = model(edges,coor)
    loss = F.mse_loss(output, label)
    loss.backward()
    train_losses.append(loss.item())
    optimiser.step()
    print("loss:" + str(loss.item()))

    if i % 1000 == 0:
      print(i, loss.item())
      torch.save(model.state_dict(), 'netstuff/model2.pth')
      torch.save(optimiser.state_dict(), 'netstuff/optimiser2.pth')
      torch.save(train_losses, 'netstuff/train_losses2.pth')
    if i ==60001:
      break