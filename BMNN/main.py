import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model import PointNet
from utils.train import train
from utils.test import test
from utils.data.dataset import Dataset
from utils.data.dataloader import Dataloader

import csv
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--path', type=str, default='/Users/shohei/PycharmProjects/BMNN_exp/dataset', help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")


opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = opt.path
opt.n = 1000
opt.d = 10
opt.lr = 0.001


def main(opt):
    train_dataset = Dataset(opt.dataroot, True)
    train_dataloader = Dataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=False, num_workers=2)

    test_dataset = Dataset(opt.dataroot, False)
    test_dataloader = Dataloader(test_dataset, batch_size=opt.batchSize, \
                                      shuffle=False, num_workers=2)

    net = PointNet(d=opt.d, feature_transform=opt.feature_transform)
    net.double()
    print(net)

    criterion = nn.CosineSimilarity(dim=2)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    with open('train.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["train_loss", "train_gain", "baseline_loss", "baseline_gain"])

    with open('test.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["test_loss", "test_gain", "baseline_loss", "baseline_gain"])

    start = time.time()

    for epoch in range(0, opt.niter):
        train(epoch, train_dataloader, net, criterion, optimizer, opt)
        test(test_dataloader, net, criterion, optimizer, opt)

    elapsed_time = time.time() - start

    with open('time.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["学習時間", elapsed_time])


if __name__ == "__main__":

    """
    input1 = torch.randn(4, 2, 3)
    input2 = torch.randn(4, 2, 3)
    cos = nn.CosineSimilarity(dim=2)
    output = cos(input1, input2)
    print(output.shape)
    avg = nn.AvgPool1d(2)
    output = torch.unsqueeze(output, dim=1)
    print(output.shape)
    print(output)
    output = avg(output)
    print(output)
    print(output.shape)
    output = torch.squeeze(output, dim=1)
    print(output.shape)
    criterion = nn.Sequential(
        nn.CosineSimilarity(dim=2),
        torch.unsqueeze(dim=1),
        nn.AvgPool1d(2),
        torch.squeeze(dim=1)
    )
    print(criterion(input1, input2).shape)
    """

    main(opt)

