from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        # x.shape (32, 3, 2500)
        batchsize = x.size()[0]
        # batchsize 3
        x = F.relu(self.bn1(self.conv1(x)))
        # x.shape (32, 64, 2500)
        x = F.relu(self.bn2(self.conv2(x)))
        # x.shape (32, 128, 2500)
        x = F.relu(self.bn3(self.conv3(x)))
        # x.shape (32, 1024, 2500)
        x = torch.max(x, 2, keepdim=True)[0]
        # x.shape (32, 1024, 1)
        x = x.view(-1, 1024)
        # x.shape (32, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        # x.shape (32, 512)
        x = F.relu(self.bn5(self.fc2(x)))
        # x.shape (32, 256)
        x = self.fc3(x)
        # x.shape (32, 9)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        # iden [1., 0., 0., 0., 1., 0., 0., 0., 1.] * 32
        # iden.shape (32, 9)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        # x.shape (32, 9)
        x = x.view(-1, 3, 3)
        # x.shape (32, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # x.shape (32, 3, 2500)
        n_pts = x.size()[2]
        # n_pts 2500
        trans = self.stn(x)
        # trans.shape (32, 3, 3)
        x = x.transpose(2, 1)
        # x.shape (32, 2500, 3)
        x = torch.bmm(x, trans)
        # x.shape (32, 2500, 3)
        x = x.transpose(2, 1)
        # x.shape (32, 3, 2500)
        x = F.relu(self.bn1(self.conv1(x)))
        # x.shape (32, 64, 2500)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        # pointfeat.shape (32, 64, 2500)
        x = F.relu(self.bn2(self.conv2(x)))
        # x.shape (32, 128, 2500)
        x = self.bn3(self.conv3(x))
        # x.shape (32, 1024, 2500)
        x = torch.max(x, 2, keepdim=True)[0]
        # x.shape (32, 1024, 1)
        x = x.view(-1, 1024)
        # x.shape (32, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            # x.shape (32, 1024, 2500)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet(nn.Module):
    def __init__(self, d, feature_transform=False):
        super(PointNet, self).__init__()
        self.d = d
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.d, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(2,1).contiguous()
        # x.shape (32, 3, 2500)
        batchsize = x.size()[0]
        # batchsize 32
        n_pts = x.size()[2]
        # n_pts 2500
        x, trans, trans_feat = self.feat(x)
        # x.shape (32, 1088, 2500)
        # trans.shape (32, 3, 3)
        # trans_feat None
        x = F.relu(self.bn1(self.conv1(x)))
        # x.shape (32, 512, 2500)
        x = F.relu(self.bn2(self.conv2(x)))
        # x.shape (32, 256, 2500)
        x = F.relu(self.bn3(self.conv3(x)))
        # x.shape (32, 128, 2500)
        x = self.conv4(x)
        # x.shape (32, 3, 2500)
        x = x.transpose(2,1).contiguous()
        # x.shape (32, 2500, 3)
        return x, trans, trans_feat