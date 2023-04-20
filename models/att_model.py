import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss

from utils.utils import initialize_weights

class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes

# My Net
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=256, D=512, dropout=False, n_classes=21, att=False):
        super(Attn_Net_Gated, self).__init__()
        self.att = att
        self.attention_a = [nn.Linear(L, D),
                            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.5))
            self.attention_b.append(nn.Dropout(0.5))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)  # W

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)  # 点乘
        A = self.attention_c(A)  # N x n_classes => num_patch × n_classes
        if self.att:
            A = A.mean(dim=1)
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""

class CLAM_SB_Reg(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=21,
                 freeze=False):
        super(CLAM_SB_Reg, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)  # size[1]=512, n_classes=10
        self.n_classes = n_classes

        initialize_weights(self)
        if freeze:
            self.attention_net.requires_grad_(False)

    def relocate(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
       # self.instance_classifiers = self.instance_classifiers.to(device)

    def forward(self, h,attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        logits = self.classifiers(M)  # 返回分类层的输出 维度=n_classes
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)
        # if instance_eval:
        #     results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
        #                     'inst_preds': np.array(all_preds)}
        # else:
        #     results_dict = {}
        return logits,M


class CLAM_SB_Reg_NN_Pool(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=1,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, freeze=False, N=100,
                 size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
                 ):
        super(CLAM_SB_Reg_NN_Pool, self).__init__()
        self.size_dict = size_dict
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1] * 2, n_classes)
        self.n_classes = n_classes
        self.N = N

        initialize_weights(self)
        if freeze:
            self.attention_net.requires_grad_(False)

    def relocate(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)


    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N

        if h.shape[0] > self.N * 2:
            idxs = torch.argsort(A[0])
            low_n_idxs = idxs[:self.N]
            high_n_idxs = idxs[-self.N:]

            low_n = h[low_n_idxs].mean(axis=0)
            high_n = h[high_n_idxs].mean(axis=0)

            M = torch.cat([low_n, high_n])
            M = torch.unsqueeze(M, 0)
        else:
            M = torch.mm(A, h)
            M = torch.concat([M[0], M[0]])
            M = torch.unsqueeze(M, 0)

        if return_features:
            return M
        logits = self.classifiers(M)

        return logits,M

class CLAM_SB_Class_NN_Pool(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=1,
                 instance_loss_fn=nn.CrossEntropyLoss(), mlp=0, freeze=False, N=100,
                 size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
                 ):
        super(CLAM_SB_Class_NN_Pool, self).__init__()
        self.size_dict = size_dict
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.5))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Sequential(nn.Linear(size[1]*2, n_classes)) if mlp == 0 else nn.Sequential(nn.Linear(size[1]*2, mlp), nn.ReLU(),nn.Dropout(0.5), nn.Linear(mlp, n_classes))
        self.n_classes = n_classes
        self.N = N

        initialize_weights(self)
        if freeze:
            self.attention_net.requires_grad_(False)

    def relocate(self, device):
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)


    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N

        if h.shape[0] > self.N * 2:
            idxs = torch.argsort(A[0])
            low_n_idxs = idxs[:self.N]
            high_n_idxs = idxs[-self.N:]

            low_n = h[low_n_idxs].mean(axis=0)
            high_n = h[high_n_idxs].mean(axis=0)

            M = torch.cat([low_n, high_n])
            M = torch.unsqueeze(M, 0)
        else:
            M = torch.mm(A, h)
            M = torch.concat([M[0], M[0]])
            M = torch.unsqueeze(M, 0)

        if return_features:
            return M
        logits = self.classifiers(M)

        return logits,M


class CLAM_MB_Reg(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, mlp=0, n_classes=21,
                 ):
        super(CLAM_MB_Reg, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention)
        self.attention_net = nn.Sequential(*fc)
        if mlp == 0:
            bag_classifiers = [nn.Linear(size[1], 1) for i in
                               range(n_classes)] # use an indepdent linear layer to predict each class
        else:
            bag_classifiers = [nn.Sequential(nn.Linear(size[1], mlp), nn.ReLU(), nn.Linear(mlp, 1))
                               for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        
        self.n_classes = n_classes
        self.n_tasks = n_classes
        initialize_weights(self)
    
    def forward(self, h,  attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        del A, A_raw, h
        gc.collect()
        return logits,M

    def get_last_shared_layer(self):
        return self.attention_net[3].attention_c


class CLAM_SB_Class(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, mlp=0, n_tasks=1,
                 n_classes = 5):
        super(CLAM_SB_Class, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [2048, 512, 256]}
        size = self.size_dict[size_arg]
        
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.5))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_tasks)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_tasks)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        self.classifier =nn.Sequential(nn.Linear(size[1], n_classes)) if mlp == 0 else nn.Sequential(nn.Linear(size[1], mlp), nn.ReLU(),nn.Dropout(0.5), nn.Linear(mlp, n_classes))#64), nn.ReLU(),nn.Dropout(0.5), nn.Linear(64,nn.Linear(size[1], n_classes)
        
        self.n_tasks = n_tasks
        initialize_weights(self)
    
    def forward(self, h, return_features=False, attention_only=False):
        device = h.device
        A, h = self.attention_net(h)  # NxK
        #print(A.shape)
        A = torch.transpose(A, 1, 0)  # KxN
       # print(A.shape)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        # print(A.shape)
        # print(h.shape)
        M = torch.mm(A, h)
        #print(M.shape)
        del A,A_raw,h
        logits = self.classifier(M)
        #print(logits)
        return logits, M

