import numpy as np
import glob
import re

n_samples = 1000
ratio_test = 0.2

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

class Dataset():
    def __init__(self, path, is_train):
        # 入力ファイルのPATHのリストを取得
        baseline_paths = load_paths_from_dir(path + "/proposal")
        proposal_paths = load_paths_from_dir(path + "/proposal")
        target_paths = load_paths_from_dir(path + "/target")

        # split data
        n_samples = len(target_paths)
        train_idx, test_idx = train_test_split(n_samples, ratio_test)
        if is_train:
            target_idx = train_idx
        else:
            target_idx = test_idx

        # ファイル読み込み
        self.idx_list = target_idx
        self.baseline = [np.load(baseline_paths[idx]) for idx in target_idx]
        self.proposal = [np.load(proposal_paths[idx]) for idx in target_idx]
        self.target = [np.load(target_paths[idx]) for idx in target_idx]

    def __getitem__(self, index):
        sample_idx = self.idx_list[index]
        baseliene = self.baseline[index]
        proposal = self.proposal[index]
        target = self.target[index]
        return sample_idx, baseliene, proposal, target

    def __len__(self):
        return len(self.idx_list)