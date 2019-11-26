import torch
from torch.autograd import Variable
import numpy as np
import copy

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


def train(epoch, dataloader, net, optimizer, opt):
    train_loss = 0
    net.train()
    for i, (sample_idx, baseliene, proposal, target) in enumerate(dataloader, 0):
        net.zero_grad()

        baseline = Variable(baseliene)
        proposal = Variable(proposal)
        target = Variable(target)

        proposal, _, _ = net(proposal)

        for batch in range(target.shape[0]):
            b = {i: baseline[batch][i].tolist() for i in range(opt.n)}
            p = {i: proposal[batch][i].tolist() for i in range(opt.n)}
            t = {i: target[batch][i].tolist() for i in range(opt.n)}
            b_node_pair_list, _ = BipartiteMatching(b, t)
            p_node_pair_list, _ = BipartiteMatching(p, t)




        loss = criterion(output, target)
        train_loss += loss

        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader.dataset)
    print('Train set: Average loss: {:.4f}'.format(train_loss))