import numpy as np
from torch.utils.data import Dataset


class Feeder(Dataset):

    def __init__(
        self,
        data_path,
        label_path,
        severity_path,
        rrb_path,
        debug=False
    ):

        self.data = np.load(data_path)
        self.label = np.load(label_path)
        self.severity = np.load(severity_path)
        self.rrb = np.load(rrb_path)
        self.debug = debug
        self.sample_name = np.arange(len(self.label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        action_label = self.label[index]
        severity_label = self.severity[index]
        rrb_label = self.rrb[index]

        return (
            data_numpy,
            action_label,
            severity_label,
            rrb_label,
            index
        )

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [
            l in rank[i, -top_k:]
            for i, l in enumerate(self.label)
        ]

        return (
            sum(hit_top_k) * 1.0 / len(hit_top_k)
        )