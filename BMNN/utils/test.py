import torch
from torch.autograd import Variable
import numpy as np

def test(dataloader, net, optimizer, opt):
    test_loss = 0
    net.eval()
    for i, (sample_idx, baseliene, proposal, target) in enumerate(dataloader, 0):
        net.zero_grad()

        baseline = Variable(baseliene)
        proposal = Variable(proposal)
        target = Variable(target)

        proposal, _, _ = net(proposal)



        test_loss += criterion(output, target).item()

        # testデータの予測結果とラベルを保存
        OutputDir = "/Users/shohei/work/AQinference/data/output"
        np.save(OutputDir+"/fnn"+str(sample_idx.numpy()[0]), output.detach().numpy()[0].reshape(10))

    test_loss /= len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))