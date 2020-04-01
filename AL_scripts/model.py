"""
Script to get the model to use for the active learning
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

def get_model(name):
    if name.upper() == "CIFAR10":
        return Net
    if name.upper() == "XRAY":
        return Net2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(3, 6, 5).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(self.device)
        self.fc1 = nn.Linear(16 * 5 * 5, 120).to(self.device)
        self.fc2 = nn.Linear(120, 84).to(self.device)
        self.fc3 = nn.Linear(84, 10).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        e1 = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = nn.Conv2d(1, 6, 5).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.conv2 = nn.Conv2d(6, 16, 5).to(self.device)
        self.fc1 = nn.Linear(16*28*28, 120).to(self.device)
        self.fc2 = nn.Linear(120, 84).to(self.device)
        self.fc3 = nn.Linear(84, 2).to(self.device)

    # def input_layer(self, x):
    #     return nn.Linear(x.shape[0], 120).to(self.device)

    def forward(self, x):
        x = x.to(self.device).float()
        x = self.pool(F.relu(self.conv1(x))).float()
        x = self.pool(F.relu(self.conv2(x))).float()
        x = x.view(-1, x.size()[1]*x.size()[2]*x.size()[3]).float()
        # x = self.input_layer(x)
        x = F.relu(self.fc1(x)).float()
        x = F.relu(self.fc2(x)).float()
        x = self.fc3(x).float()
        return x