import torch
import torch.nn as nn
import torch.nn.functional as F
from module.view_estimator import View_Estimator
from module.cls import Classification
from module.att_fusion import AttFusion


class Cogsl(nn.Module):
    def __init__(self, num_node, num_feature, cls_hid_1, num_class, gen_hid, fu_hid,
                 com_lambda_v1, com_lambda_v2, cls_dropout, ve_dropout, pyg, name):
        super(Cogsl, self).__init__()
        self.cls = Classification(num_feature, cls_hid_1, num_class, cls_dropout, pyg)
        self.ve = View_Estimator(num_node, num_feature, gen_hid, com_lambda_v1, com_lambda_v2, ve_dropout, pyg, name)
        self.attfusion = AttFusion(num_node, fu_hid)

    def get_view(self, data):
        new_v1, new_v2 = self.ve(data)
        return new_v1, new_v2

    def get_cls_loss(self, v1, v2, feat):
        prob_v1 = self.cls(feat, v1, "v1")
        prob_v2 = self.cls(feat, v2, "v2")
        logits_v1 = torch.log(prob_v1 + 1e-8)
        logits_v2 = torch.log(prob_v2 + 1e-8)
        return logits_v1, logits_v2, prob_v1, prob_v2

    def get_v_cls_loss(self, v, feat):
        logits = torch.log(self.cls(feat, v, "v") + 1e-8)
        return logits

    def get_fusion(self, v1, v2):
        views = torch.stack([v1.to_dense(), v2.to_dense()], dim=1)
        v = self.attfusion(views)
        return v.to_sparse()