# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch.nn as nn

from utils.utils import initialize_weights


class MLP(nn.Module):

    def __init__(self, L=1024, D=512, dropout=False, n_classes=1):
        super(MLP, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)]

        # self.module.append(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, n_classes))

        self.module = nn.Sequential(*self.module)
        initialize_weights(self)

    def forward(self, x):
        return self.module(x),None  # N x n_classes


class MLP_gene(nn.Module):

    def __init__(self, L=1024, D=512, dropout=False, n_classes=1):
        super(MLP_gene, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)]

        # self.module.append(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, n_classes))

        self.module = nn.Sequential(*self.module)
        initialize_weights(self)

    def forward(self, x):
        return self.module(x)  # N x n_classes

class MultiScaleMLP(nn.Module):

    def __init__(self, L=2048, D=1024, n_classes=1):
        super(MultiScaleMLP, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_classes)]

        self.module = nn.Sequential(*self.module)
        initialize_weights(self)

    def forward(self, x):
        return self.module(x)  # N x n_classes


if __name__ == "__main__":
    pass
