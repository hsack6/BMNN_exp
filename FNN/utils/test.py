import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import copy
import csv

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def Matching(n_list_tup, t_list_tup, n, t):
    pair_list = []
    decided_n = set()
    decided_t = set()
    while True:
        for idx in range(len(n)):
            matched_n, matched_t = n[idx], t[idx]
            if matched_n in decided_n or matched_t in decided_t:
                continue
            pair_list.append((n_list_tup[matched_n][0], t_list_tup[matched_t][0]))
            decided_n.add(matched_n)
            decided_t.add(matched_t)
            if len(decided_n) == len(n_list_tup):
                break
            if len(decided_t) == len(t_list_tup):
                decided_t = set()
                break
        if len(decided_n) == len(n_list_tup):
            break
    return pair_list


def BipartiteMatching(new_vec_dic, teacher_vec_dic):
    # sort済みのタプルのリスト[(node_id, vector)]
    n_list_tup = sorted(new_vec_dic.items(), key=lambda x: x[0])
    t_list_tup = sorted(teacher_vec_dic.items(), key=lambda x: x[0])
    # 二部グラフ全てのsimilarityを計算
    N = np.array([n_v for n, n_v in n_list_tup])
    T = np.array([t_v for t, t_v in t_list_tup])
    normN = np.sqrt(np.sum(N * N, axis=1))
    normT = np.sqrt(np.sum(T * T, axis=1))
    similarity_matrix = np.dot(N / normN.reshape(-1, 1), (T / normT.reshape(-1, 1)).T)
    # similarityのsort
    n, t = np.unravel_index(np.argsort(-similarity_matrix.reshape(-1)), similarity_matrix.shape)
    # Greedy Matching
    node_pair_list = Matching(copy.copy(n_list_tup), copy.copy(t_list_tup), n.tolist(), t.tolist())
    return node_pair_list, similarity_matrix

def test(dataloader, net, criterion, optimizer, opt):
    test_loss = 0
    baseline_gain = 0
    net.eval()
    for i, (sample_idx, baseliene, proposal, target) in enumerate(dataloader, 0):
        net.zero_grad()

        baseline = Variable(baseliene)
        proposal = Variable(proposal)
        target = Variable(target)

        proposal = net(proposal)

        transformed_target = torch.Tensor(target.shape[0], opt.n, opt.d)
        b_similarity = 0
        for batch in range(opt.batchSize):
            p = {i: proposal[batch][i].tolist() for i in range(opt.n)}
            t = {i: target[batch][i].tolist() for i in range(opt.n)}
            b = {i: baseline[batch][i].tolist() for i in range(opt.n)}
            p_node_pair_list, _ = BipartiteMatching(p, t)
            b_node_pair_list, b_similarity_matrix = BipartiteMatching(b, t)

            for tpl in sorted(p_node_pair_list, key=lambda x: x[0]):
                transformed_target[batch][tpl[0]] = target[batch][tpl[1]]

            b_score = 0
            for i in range(opt.n):
                b_score += b_similarity_matrix[b_node_pair_list[i]]
            b_score /= opt.n

            b_similarity += b_score

        b_similarity /= opt.batchSize
        target = transformed_target
        gain = nn.CosineSimilarity(dim=2)(proposal, target)
        gain = gain.mean()
        loss = -1 * gain.log()
        print("test_loss:" + str(loss) + ", test_gain:" + str(gain) + ", baseline_gain:" + str(b_similarity))
        test_loss += loss
        baseline_gain += b_similarity

    test_loss /= (len(dataloader.dataset) / opt.batchSize)
    baseline_gain /= (len(dataloader.dataset) / opt.batchSize)
    test_gain = torch.exp(-1 * test_loss)
    baseline_loss = -1 * np.log(baseline_gain)
    print('Test set: Average loss: {:.4f}, Average gain: {:.4f}, Baseline loss: {:.4f}, Baseline gaine: {:.4f}'.format(test_loss.item(), test_gain.item(), baseline_loss, baseline_gain))

    log = [test_loss.item(), test_gain.item(), baseline_loss, baseline_gain]
    with open('test.csv', 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(log)