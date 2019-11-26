import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from model import FNN
from utils.train import train
from utils.test import test
from utils.data.dataset import AQDataset
from utils.data.dataloader import AQDataloader

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
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


def main(opt):
    train_dataset = AQDataset(opt.dataroot, True)
    train_dataloader = AQDataloader(train_dataset, batch_size=opt.batchSize, \
                                      shuffle=True, num_workers=2)

    test_dataset = AQDataset(opt.dataroot, opt.L, False)
    test_dataloader = AQDataloader(test_dataset, batch_size=opt.batchSize, \
                                     shuffle=False, num_workers=2)

    opt.annotation_dim = train_dataset[0][1].shape[2]
    opt.n_edge_types = train_dataset.n_edge_types
    opt.n_node = train_dataset.n_node

    net = FNN(opt, hidden_state=opt.state_dim*2)
    net.double()
    print(net)

    criterion = nn.MSELoss()


    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

    for epoch in range(0, opt.niter):
        train(epoch, train_dataloader, net, criterion, optimizer, opt)
        test(test_dataloader, net, criterion, optimizer, opt)


if __name__ == "__main__":
    main(opt)

