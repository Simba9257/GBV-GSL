import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv


def knn(feat, num_node, k, data_name, view_name):
    adj = np.zeros((num_node, num_node), dtype=np.int64)
    dist = cos(feat)
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    #adj[np.arange(dataset.num_node).repeat(k + 1), col] = 1
    adj[np.arange(num_node).repeat(k + 1), col] = 1
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_knn.npz", adj)


def adj_(adj, data_name, view_name):
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_adj.npz", adj)


def diff(adj, alpha, data_name, view_name):   
    d = np.diag(np.sum(adj, 1))                                    
    dinv = fractional_matrix_power(d, -0.5)                       
    at = np.matmul(np.matmul(dinv, adj), dinv)                      
    adj = alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at))   
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_diff.npz", adj)

data_name = "chameleon"
view_name = "v2"  # v1 or v2
view_type = "diff"  # knn adj diff

#adj = sp.load_npz("./"+data_name+"/ori_adj.npz")
adj = pickle.load(open(f'./{data_name}/{data_name}_adj.pkl', 'rb'))
print('adj',adj)
num_node = adj.shape[0]
#feat = sp.load_npz("./"+data_name+"/feat.npz")
feat = pickle.load(open(f'./{data_name}/{data_name}_features.pkl', 'rb'))
print('feat',feat)
a = adj.A
if a[0, 0] == 0:
    #a += np.eye(dataset.num_node)
    a += np.eye(num_node)
    print("self-loop!")
adj = a
if view_type == "knn":  # set k
    #knn(feat, num_node, k, data_name, view_name)
    knn(feat, num_node, 2, data_name, view_name)
elif view_type == "adj":
    adj_(adj, data_name, view_name)
elif view_type == "diff":  # set alpha: 0~1
    #diff(adj, alpha, data_name, view_name)
    diff(adj, 0.5, data_name, view_name)
