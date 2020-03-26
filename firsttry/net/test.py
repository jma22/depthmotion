from pytorch_lightning import Trainer
from vanillanet import messNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
from classes import data2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchsummary import summary



data_path = '../datagen/data2/cube_data.pickle'
truth_path = '../datagen/data2/cube_truth.pickle'
net_path = './netstuff/model.pth'
loss_path = './netstuff/test_losses.pth'

train_data = data2(list(range(0,1000)),data_path,truth_path)

train_loader = DataLoader(train_data, batch_size=1,
                          shuffle=True, num_workers=4, pin_memory=True)


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
model = messNet().to(device)
# model = messNet()
model.eval()
test_loss, correct = 0, 0
model.load_state_dict(torch.load(net_path))

# losses = torch.load(loss_path)
# test_loss = torch.load(loss_path)
test_loss = []
with torch.no_grad():
  for i, (edges,coor,label) in enumerate(train_loader):
    print(i)
    if i <60000:
      pass
    else:
      edges = edges.to(device=device, non_blocking=True)
      coor = coor.to(device=device, non_blocking=True)
      label = label.to(device=device, non_blocking=True)
      output = model(edges,coor)
      loss = F.mse_loss(output, label)
      test_loss.append(loss.item())
      print(loss.item)
    if i ==70000:
      break
torch.save(test_loss, 'netstuff/test_losses.pth')
#see parameters
# for param in model.parameters():
#   print(param)
  # print(param.data)
# print(model.parameters())
# summary(model,input_size=[( 1, 2), (1, 6)])

#print losses
print(len(test_loss))
for i in test_loss:
  print(i.item())
# plt.plot(test_loss[-500:-1])
# plt.savefig('netstuff/foo.png')