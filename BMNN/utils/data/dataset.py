import numpy as np
import glob
import re


def load_paths_from_dir(dir_path):
    # dir 以下のファイル名のリストを取得
    path_list = glob.glob(dir_path + "/*")
    # ソート (ゼロ埋めされていない数字の文字列のソート)
    path_list = np.array(sorted(path_list, key=lambda s: int(re.findall(r'\d+', s)[-1])))
    return path_list

def train_test_split(n_samples, ratio_test):
    idx = list(range(n_samples))
    n_test = int(n_samples * ratio_test)
    return idx[:n_samples - n_test], idx[-n_test:]

class AQDataset():
    def __init__(self, path, is_train):
        # 入力ファイルのPATHのリストを取得
        data_paths = load_paths_from_dir(path)
        print(data_paths)
        aaa

        # split data
        n_samples = len(label_return_paths)
        train_idx, test_idx = train_test_split(n_samples, ratio_test)
        if is_train:
            target_idx = train_idx
        else:
            target_idx = test_idx

        # ファイル読み込み(scipy cooはDataLoaderのサポート外なので変換する)
        self.idx_list = target_idx
        #self.attribute = [np.array(mmread(attribute_paths[idx]).todense()).reshape((adj_shape[0], L, attribute_dim)) for idx in target_idx]  # scipy疎行列をnumpy密行列に変換してからreshape
        self.adjacency = [in_out_generate(coo_scipy2coo_numpy(mmread(adjacency_paths[idx]), max_nnz_am), adj_shape[0]) for idx in target_idx]  # scipy疎行列をnumpy疎行列に変換し、GGNN用にinとoutを生成する
        self.label_edge = [coo_scipy2coo_numpy(mmread(label_edge_paths[idx]), max_nnz_label_edge) for idx in target_idx]  # scipy疎行列をnumpy疎行列に変換
        #self.label_attribute = [np.load(label_attribute_paths[idx]) for idx in target_idx]
        self.label_lost = [np.load(label_lost_paths[idx]) for idx in target_idx]
        self.label_return = [np.load(label_return_paths[idx]) for idx in target_idx]

        # PrimaryAttribute(fixかmobileか = dim0, dim1)を捨てる。label_attributeもPM2.5のみ。
        self.attribute = [np.array(mmread(attribute_paths[idx]).todense()).reshape((adj_shape[0], L, attribute_dim))[:,:,2:].reshape(adj_shape[0], L, attribute_dim-2) for idx in target_idx]
        self.label_attribute = [np.load(label_attribute_paths[idx])[:,2].reshape(adj_shape[0], 1) for idx in target_idx]

        # 入力グラフの統計量
        self.L = L
        self.n_node = adj_shape[0]
        self.n_edge_types = adj_shape[1] // adj_shape[0]


    def __getitem__(self, index):
        sample_idx = self.idx_list[index] + self.L
        annotation = self.attribute[index]
        am = self.adjacency[index]
        label_edge = self.label_edge[index]
        label_attribute = self.label_attribute[index]
        label_lost = self.label_lost[index]
        label_return = self.label_return[index]
        return sample_idx, annotation, am, label_edge, label_attribute, label_lost, label_return

    def __len__(self):
        return len(self.idx_list)