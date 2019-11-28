import numpy as np
import copy

n = 1000
d = 10
n_sample = 100

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

seed_array = np.array([np.random.randn(d), np.random.rand(d)]).transpose(1, 0)
eps_array = np.array([np.random.randn(d), np.random.rand(d)]).transpose(1, 0)

b_score_list = []
for s in range(n_sample):
    # TODO: サンプルごとに異なるシードで学習できるのか検証。現実的にサンプル間（時系列間）の統計量はそんなに変わらなさそう。

    node_attribute_init = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            node_attribute_init[i][j] = np.random.normal(seed_array[j][0], seed_array[j][1])
    baseline = (node_attribute_init - np.min(node_attribute_init)) / (
            np.max(node_attribute_init) - np.min(node_attribute_init))
    proposal = baseline

    target = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            target[i][j] = np.random.normal(seed_array[j][0] + eps_array[j][0], seed_array[j][1] + eps_array[j][1])
    target = (target - np.min(target)) / (np.max(target) - np.min(target))

    np.save("dataset/baseline/" + str(s) + ".npy", baseline)
    np.save("dataset/proposal/" + str(s) + ".npy", proposal)
    np.save("dataset/target/" + str(s) + ".npy", target)

    b = {i: baseline[i].tolist() for i in range(n)}
    #p = {i: proposal[i].tolist() for i in range(n)}
    t = {i: target[i].tolist() for i in range(n)}
    b_node_pair_list, b_similarity_matrix = BipartiteMatching(b, t)
    #p_node_pair_list, p_similarity_matrix = BipartiteMatching(p, t)

    b_score = 0
    #p_score = 0
    for i in range(n):
        b_score += b_similarity_matrix[b_node_pair_list[i]]
        #p_score += p_similarity_matrix[p_node_pair_list[i]]
    b_score = b_score / n
    #_score = p_score / n

    b_score_list.append(b_score)

    print(s)
print(np.array(b_score_list).mean())