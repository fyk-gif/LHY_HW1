# -*— coding: utf-8 -*-
# @Time : 2021/6/15 10:09
# @Author : FYK

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def load_data(data_root, target_only=True):
    features = []

    data_files = os.listdir(data_root)
    test_file = os.path.join(data_root, data_files[0])
    train_file = os.path.join(data_root, data_files[1])
    with open(train_file, "r") as train_f:
        with open(test_file, "r") as test_f:
            train_data = list(csv.reader(train_f))
            train_data = np.array(train_data[1:])[:, 1:].astype(float)
            np.random.shuffle(train_data)

            val_data = train_data[:200]
            train_data = train_data[200:]

            test_data = list(csv.reader(test_f))
            test_data = np.array(test_data[1:])[:, 1:].astype(float)

            if target_only:
                features = list(range(93))
            else:
                #  TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
                pass

            val_label = val_data[:, -1]
            val_data = val_data[:, features]

            train_label = train_data[:, -1]
            train_data = train_data[:, features]

            test_data = test_data[:, features]

    return train_data, train_label, val_data, val_label, test_data


class CovidDataset(Dataset):
    def __init__(self, data_x, data_y=None):
        super(CovidDataset, self).__init__()

        self.data_x = torch.from_numpy(data_x).float()
        if data_y is not None:
            self.data_y = torch.from_numpy(data_y).float()
        else:
            self.data_y = None

    def __getitem__(self, item):
        return self.data_x[item], self.data_y[item]

    def __len__(self):
        return len(self.data_x)


if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x = load_data("data")
    print(len(train_x))
    print(train_x[0].shape)

    print(len(val_x))
    print(val_x[0].shape)

    # train_dataset = CovidDataset(train_x, train_y)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    # train_loader = iter(train_loader)
    # data = next(train_loader)
    # x, y = data
    # print(x, y)