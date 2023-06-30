import csv
import time

import numpy as np
import random
import torch
import torch.nn.functional as F
from utils import load_data, accuracy, load_data2
from params import set_params
from cogsl import Cogsl
import warnings
from copy import deepcopy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

args = set_params()
if torch.cuda.is_available() and args.gpu!=-1:
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")


if args.add:
    own_str = args.dataset+"_add_"+str(args.ratio)+str(args.flag)
elif args.dele:
    own_str = args.dataset+"_dele_"+str(args.ratio)+str(args.flag)
elif args.ptb_feat:
    own_str = args.dataset+"_ptb_feat_"+str(args.ratio)
else:
    own_str = args.dataset
print(own_str)


## random seed ##
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)


def evaluation(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels)
    accuracy = correct.item() / len(labels)

    return accuracy


def gen_mima(logits, label):
    preds = torch.argmax(logits, dim=1)
    test_f1_macro = f1_score(label.cpu(), preds.cpu(), average='macro')
    test_f1_micro = f1_score(label.cpu(), preds.cpu(), average='micro')
    return test_f1_macro, test_f1_micro


class TrainFlow():
    def __init__(self, own_str, device):
        data_path = "../dataset/"
        ptb_path = "../ptb/"
        if args.dataset == "cora" or args.dataset == "texas" or args.dataset == "cornell" or args.dataset == "wisconsin":
            self.data = load_data2(data_path, ptb_path, args.dataset, args).to(device)
        else:
            self.data = load_data(data_path, ptb_path, args.dataset, args).to(device)
        self.main_model = Cogsl(self.data.num_node, self.data.num_feature, args.cls_hid_1, self.data.num_class,
                                     args.gen_hid, args.fu_hid, args.com_lambda_v1, args.com_lambda_v2,
                           args.cls_dropout, args.ve_dropout, args.pyg, args.dataset)

        self.opti_ve = torch.optim.Adam(self.main_model.ve.parameters(), lr=args.ve_lr, weight_decay=args.ve_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opti_ve, 0.99)

        self.opti_cls = torch.optim.Adam(self.main_model.cls.parameters(), lr=args.cls_lr, weight_decay=args.cls_weight_decay)
        self.opti_fusion = torch.optim.Adam(self.main_model.attfusion.parameters(), lr=args.fu_lr, weight_decay=args.fu_weight_decay)

        if torch.cuda.is_available() and args.gpu!=-1:
            print('Using CUDA')
            self.main_model.cuda()
        self.best_acc_val = 0
        self.best_loss_val = 1e9
        self.best_test = 0
        self.best_v = None
        self.best_v_cls_weight = None
        self.own_str = own_str

    def loss_acc(self, output, y):
        loss = F.nll_loss(output, y)
        acc = accuracy(output, y)
        return loss, acc

    def train_cls(self):
        new_v1, new_v2 = self.main_model.get_view(self.data)
        logits_v1, logits_v2, prob_v1, prob_v2 = self.main_model.get_cls_loss(new_v1, new_v2, self.data.x)
        curr_v = self.main_model.get_fusion(new_v1, new_v2)
        logits_v = self.main_model.get_v_cls_loss(curr_v, self.data.x)
          
        views = [curr_v, new_v1, new_v2]
        
        loss_v1, _ = self.loss_acc(logits_v1[self.data.idx_train], self.data.y[self.data.idx_train])
        loss_v2, _ = self.loss_acc(logits_v2[self.data.idx_train], self.data.y[self.data.idx_train])
        loss_v, _ = self.loss_acc(logits_v[self.data.idx_train], self.data.y[self.data.idx_train])
        return args.cls_coe * loss_v + (loss_v1 + loss_v2) * (1 - args.cls_coe) / 2, views

    def train(self):
        for epoch in range(args.main_epoch):

            # view estimator/interaction training
            for inner_ne in range(args.inner_ne_epoch):
                self.main_model.train()
                self.opti_ve.zero_grad()
                cls_loss, _ = self.train_cls()
                with torch.autograd.detect_anomaly():
                    cls_loss.backward()
                self.opti_ve.step()
            self.scheduler.step()

            # classifiers training
            for inner_cls in range(args.inner_cls_epoch):
                self.main_model.train()
                self.opti_cls.zero_grad()
                cls_loss, _ = self.train_cls()
                with torch.autograd.detect_anomaly():
                    cls_loss.backward()
                self.opti_cls.step()

            # attfusion training
            for inner_fu in range(args.inner_fu_epoch):
                self.main_model.train()
                self.opti_fusion.zero_grad()
                cls_loss, _ = self.train_cls()
                cls_loss.backward()
                self.opti_fusion.step()


            ## validation ##
            self.main_model.eval()
            _, views = self.train_cls()
            logits_v_val = self.main_model.get_v_cls_loss(views[0], self.data.x)
            loss_val, acc_val = self.loss_acc(logits_v_val[self.data.idx_val], self.data.y[self.data.idx_val])
            if acc_val >= self.best_acc_val and self.best_loss_val > loss_val:
                print("better v!")
                self.best_acc_val = max(acc_val, self.best_acc_val)
                self.best_loss_val = loss_val
                self.best_v_cls_weight = deepcopy(self.main_model.cls.encoder_v.state_dict())
                self.best_v = views[0]
            print("EPOCH ",epoch, "\tCUR_LOSS_VAL ", loss_val.data.cpu().numpy(), "\tCUR_ACC_Val ", acc_val.data.cpu().numpy(), "\tBEST_ACC_VAL ", self.best_acc_val.data.cpu().numpy())

        with torch.no_grad():
            self.main_model.cls.encoder_v.load_state_dict(self.best_v_cls_weight)
            self.main_model.eval()
                           
            probs = self.main_model.cls.encoder_v(self.data.x, self.best_v)
            test_f1_macro, test_f1_micro = gen_mima(probs[self.data.idx_test], self.data.y[self.data.idx_test])
            print("Test_Macro: ", test_f1_macro, "\tTest_Micro: ", test_f1_micro)

            """
            outFile = open('./results/{}_PerformMetrics.csv'.format(args.dataset), 'a+', newline='')
            writer = csv.writer(outFile, dialect='excel')
            results = [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]

            results.append(args)

            results.append(test_f1_macro)
            results.append(test_f1_micro)
            writer.writerow(results)
            """


if __name__ == '__main__':

    train = TrainFlow(own_str, device)
    train.train()



