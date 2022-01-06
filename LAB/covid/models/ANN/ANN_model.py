import torch 
import torch.nn as nn

from ann_utils import swish

class SimpleNet(nn.Module):
  
  def __init__(self):
    super().__init__()

    self.fc1 = nn.Linear(26,16)
    self.fc2 = nn.Linear(16,8)
    self.fc3 = nn.Linear(8,4)
    self.fc4 = nn.Linear(4,1)


  def forward(self, x):

    x = swish(self.fc1(x))
    x = swish(self.fc2(x))
    x = swish(self.fc3(x))
    x = torch.relu(self.fc4(x))

    return x


# Model with Xavier Innit torch.nn.init.xavier_uniform_(self.net[1].weight)
# or He nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
class SimpleNetHe(nn.Module):
  
  def __init__(self):
    super().__init__()

    self.fc1 = nn.Linear(26,16)
    self.fc2 = nn.Linear(16,8)
    self.fc3 = nn.Linear(8,4)
    self.fc4 = nn.Linear(4,1)

    #torch.nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
    torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')


  def forward(self, x):

    x = swish(self.fc1(x))
    x = swish(self.fc2(x))
    x = swish(self.fc3(x))
    x = torch.relu(self.fc4(x))

    return x

# He nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
# Increasing the size to reduce overfit 
class SimpleNetHeLarge(nn.Module):
  # added BatchNorm1D
  def __init__(self):
    super().__init__()

    self.fc1 = nn.Linear(26,16)
    self.b1 = nn.BatchNorm1d(16)
    self.fc2 = nn.Linear(16,8)
    self.b2 = nn.BatchNorm1d(8)
    self.fc3 = nn.Linear(8,4)
    self.b3 = nn.BatchNorm1d(4)
    self.fc4 = nn.Linear(4,1)

    torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

  def forward(self, x):

    x = swish(self.fc1(x))
    x = self.b1(x)
    x = swish(self.fc2(x))
    x = self.b2(x)
    x = swish(self.fc3(x))
    x = self.b3(x)
    x = torch.relu(self.fc4(x))

    return x