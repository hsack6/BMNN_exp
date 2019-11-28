from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, d, n):
        super(FNN, self).__init__()
        self.d = d
        self.n = n
        self.fc1 = torch.nn.Linear(self.d, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 1024)
        self.fc4 = torch.nn.Linear(1024, 512)
        self.fc5 = torch.nn.Linear(512, 256)
        self.fc6 = torch.nn.Linear(256, 128)
        self.fc7 = torch.nn.Linear(128, self.d)
        #self.bn1 = nn.BatchNorm1d(self.n)
        #self.bn2 = nn.BatchNorm1d(self.n)
        #self.bn3 = nn.BatchNorm1d(self.n)
        #self.bn4 = nn.BatchNorm1d(self.n)
        #self.bn5 = nn.BatchNorm1d(self.n)
        #self.bn6 = nn.BatchNorm1d(self.n)

    def forward(self, x):
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x