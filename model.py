# -*— coding: utf-8 -*-
# @Time : 2021/6/15 10:09
# @Author : FYK

import torch
import torch.nn as nn


class Linearmodel(nn.Module):
    def __init__(self, in_dim=93):
        super(Linearmodel, self).__init__()

        self.layer1 = nn.Linear(in_dim, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.outlayer = nn.Linear(128, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.outlayer(out).squeeze(1)

        return out


if __name__ == "__main__":
    net = Linearmodel(in_dim=93)
    in_data = torch.randn(10, 93)
    out_data = net(in_data)
    print(out_data)