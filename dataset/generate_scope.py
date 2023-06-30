import pickle

import numpy as np
import scipy.sparse as sp
import torch


def get_khop_indices(k, view):
    view = (view.A > 0).astype("int32")
    view_ = view
    for i in range(1, k):
        view_ = (np.matmul(view_, view.T) > 0).astype("int32")
    view_ = torch.tensor(view_).to_sparse()
    return view_.indices()


def topk(k, adj):
    pos = np.zeros(adj.shape)
    for i in range(adj.shape[0]):
        one = adj.toarray()[i].nonzero()[0]
        if len(one) > k:
            oo = np.argsort(-adj.toarray()[i, one])
            sele = one[oo[:k]]
            pos[i, sele] = adj.toarray()[i, sele]
        else:
            pos[i, one] = adj.toarray()[i, one]
    return pos


#####################
## get k-hop scope ##
## take citeseer   ##
#####################

data_name = "polblogs"
adj = sp.load_npz("./{}/v1_adj.npz".format(data_name))
#adj = pickle.load(open(f'./{data_name}/{data_name}_adj.pkl', 'rb'))
indice = get_khop_indices(1, adj)
torch.save(indice, "./{}/v1_1.pt".format(data_name))

#####################
## get top-k scope ##
## take citeseer   ##
#####################

data_name = "digits"
adj = sp.load_npz("./{}/v2_diff.npz".format(data_name))
kn = topk(100, adj)
kn = sp.coo_matrix(kn)
indice = get_khop_indices(1, kn)
torch.save(indice, "./{}/v2_100.pt".format(data_name))
