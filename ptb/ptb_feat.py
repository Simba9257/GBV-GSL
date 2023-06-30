import numpy as np
import scipy.sparse as sp
import pickle


data_name = "polblogs"
feat = sp.load_npz("../dataset/"+data_name+"/feat.npz").A
# texas
#feat = pickle.load(open(f'../dataset/{data_name}/{data_name}_features.pkl', 'rb')).numpy()
print(feat)

r = feat.max(1).mean()
dim = feat.shape[1]
for lam in [0.1, 0.3, 0.5]:
    eps = np.random.normal(size=feat.shape)
    noise = lam * r * eps
    feat += noise
    feat = sp.coo_matrix(feat)
    sp.save_npz(data_name+"_ptb_feat_"+str(lam)+".npz", feat)