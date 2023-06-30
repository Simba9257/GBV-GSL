import torch
import torch.nn as nn

from .base import GCN_one, GCN_one_pyg
from torch_sparse import SparseTensor
import numpy as np
import torch.nn.functional as F


class GenView(nn.Module):
    def __init__(self, num_feature, hid, com_lambda, dropout, pyg):
        super(GenView, self).__init__()
        if pyg == False:
            self.gen_gcn = GCN_one(num_feature, hid, activation=nn.ReLU())

        else:
            self.gen_gcn = GCN_one_pyg(num_feature, hid, activation=nn.ReLU())
        self.gen_mlp = nn.Linear(2 * hid, 1)
        nn.init.xavier_normal_(self.gen_mlp.weight, gain=1.414)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.com_lambda = com_lambda
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_ori, feat, v_indices, num_node):
        emb = self.gen_gcn(feat, v_ori)

        f1 = emb[v_indices[0]]
        f2 = emb[v_indices[1]]
        ff = torch.cat([f1, f2], dim=-1)
        temp = self.gen_mlp(self.dropout(ff)).reshape(-1)

        z_matrix = torch.sparse.FloatTensor(v_indices, temp, (num_node, num_node))
        pi = torch.sparse.softmax(z_matrix, dim=1)
        gen_v = v_ori + self.com_lambda * pi
        return gen_v


class simGRU(nn.Module):
    def __init__(self, input_size, output_size):
        super(simGRU, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)

        self.num_node = input_size

    def forward(self, gen_v1, gen_v2):
        z1 = self.linear1(gen_v1)
        z2 = self.linear2(gen_v2)
        z = torch.sigmoid(z1 + z2)
        print(z)
        ves_star1 = (1-z) * ((gen_v1+gen_v2)/2) + z * gen_v1
        ves_star2 = (1-z) * ((gen_v1+gen_v2)/2) + z * gen_v2

        return ves_star1, ves_star2


class View_Estimator(nn.Module):
    def __init__(self, num_node, num_feature, gen_hid, com_lambda_v1, com_lambda_v2, dropout, pyg, name):
        super(View_Estimator, self).__init__()
        self.v1_gen = GenView(num_feature, gen_hid, com_lambda_v1, dropout, pyg)
        self.v2_gen = GenView(num_feature, gen_hid, com_lambda_v2, dropout, pyg)

        # gate
        self.simgru = simGRU(num_node, num_node)
        self.name = name

    def forward(self, data):
        gen_v1 = self.v1_gen(data.view1, data.x, data.v1_indices, data.num_node)
        gen_v2 = self.v2_gen(data.view2, data.x, data.v2_indices, data.num_node)

        input1 = gen_v1.to_dense()
        input2 = gen_v2.to_dense()

        ves_star1, ves_star2 = self.simgru(input1, input2)

        ves_star1 = ves_star1.to_sparse()
        ves_star2 = ves_star2.to_sparse()

        if self.name in ["texas", "cornell", "wisconsin"]:
            ves_star1 = torch.sparse.softmax(ves_star1, dim=1)
            ves_star2 = torch.sparse.softmax(ves_star2, dim=1)

        new_v1 = data.normalize(ves_star1)
        new_v2 = data.normalize(ves_star2)
        
        return new_v1, new_v2
