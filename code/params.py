import argparse
import sys

argv = sys.argv
dataset = argv[1]


def wine_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="wine")
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)    

    parser.add_argument('--name_view1', type=str, default="v1_knn")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_2")
    parser.add_argument('--indice_view2', type=str, default="v2_40")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=1.0)
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)
    parser.add_argument('--gen_hid', type=int, default=64)
    
    ## fusion
    parser.add_argument('--fu_hid', type=float, default=16)  ###
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.8)

    parser.add_argument('--fu_lr', type=float, default=0.01)
    parser.add_argument('--fu_weight_decay', type=float, default=5e-4)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=100)  # 100
    parser.add_argument('--inner_ne_epoch', type=int, default=1)
    parser.add_argument('--inner_cls_epoch', type=int, default=1)
    parser.add_argument('--inner_fu_epoch', type=int, default=1)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args


def breast_cancer_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="breast_cancer")
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6)    

    parser.add_argument('--name_view1', type=str, default="v1_knn")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_1")
    parser.add_argument('--indice_view2', type=str, default="v2_300")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)
    parser.add_argument('--com_lambda_v2', type=float, default=0.1)
    parser.add_argument('--gen_hid', type=int, default=64)
    
    ## fusion
    parser.add_argument('--fu_hid', type=float, default=16)  ###
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.5)

    parser.add_argument('--fu_lr', type=float, default=0.01)
    parser.add_argument('--fu_weight_decay', type=float, default=5e-4)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=150)
    parser.add_argument('--inner_ne_epoch', type=int, default=1)
    parser.add_argument('--inner_cls_epoch', type=int, default=5)
    parser.add_argument('--inner_fu_epoch', type=int, default=5)  ###
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args


def digits_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="digits")
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)    

    parser.add_argument('--name_view1', type=str, default="v1_adj")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_1")  # v1_1
    parser.add_argument('--indice_view2', type=str, default="v2_100")  # V2_100
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.5)
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)
    parser.add_argument('--gen_hid', type=int, default=32)
    
    ## fusion
    parser.add_argument('--fu_hid', type=float, default=16)  ###
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.5)

    parser.add_argument('--fu_lr', type=float, default=0.01)
    parser.add_argument('--fu_weight_decay', type=float, default=5e-4)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=100)
    parser.add_argument('--inner_ne_epoch', type=int, default=5)
    parser.add_argument('--inner_cls_epoch', type=int, default=10)
    parser.add_argument('--inner_fu_epoch', type=int, default=1)  ###
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args


def polblogs_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="polblogs")
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)    

    parser.add_argument('--name_view1', type=str, default="v1_adj")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_1")  # v1_1
    parser.add_argument('--indice_view2', type=str, default="v2_500")  # v2_500
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)
    parser.add_argument('--com_lambda_v2', type=float, default=1.0)
    parser.add_argument('--gen_hid', type=int, default=64)
    
    ## fusion
    parser.add_argument('--fu_hid', type=float, default=16)  ###
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.01)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.8)

    parser.add_argument('--fu_lr', type=float, default=0.01)
    parser.add_argument('--fu_weight_decay', type=float, default=5e-4)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=100)
    parser.add_argument('--inner_ne_epoch', type=int, default=1)  
    parser.add_argument('--inner_cls_epoch', type=int, default=1)
    parser.add_argument('--inner_fu_epoch', type=int, default=1)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args


def texas_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="texas")

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)

    parser.add_argument('--name_view1', type=str, default="v1_knn")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_2")
    parser.add_argument('--indice_view2', type=str, default="v2_40")

    parser.add_argument('--cls_hid_1', type=int, default=16)

    ## gen
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)
    parser.add_argument('--gen_hid', type=int, default=64)

    ## fusion
    parser.add_argument('--fu_hid', type=float, default=16)  ###

    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.8)

    parser.add_argument('--fu_lr', type=float, default=0.01)
    parser.add_argument('--fu_weight_decay', type=float, default=5e-4)

    ## iter
    parser.add_argument('--main_epoch', type=int, default=100)  # 100
    parser.add_argument('--inner_ne_epoch', type=int, default=1)
    parser.add_argument('--inner_cls_epoch', type=int, default=5)
    parser.add_argument('--inner_fu_epoch', type=int, default=1)

    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)

    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')

    parser.add_argument('--ratio', type=float, default=0.)
    parser.add_argument('--flag', type=int, default=1)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def cornell_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="cornell")

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)

    parser.add_argument('--name_view1', type=str, default="v1_knn")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_2")
    parser.add_argument('--indice_view2', type=str, default="v2_40")

    parser.add_argument('--cls_hid_1', type=int, default=16)

    ## gen
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)
    parser.add_argument('--com_lambda_v2', type=float, default=1.0)
    parser.add_argument('--gen_hid', type=int, default=64)

    ## fusion
    parser.add_argument('--fu_hid', type=float, default=16)  ###

    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.1)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.8)

    parser.add_argument('--fu_lr', type=float, default=0.01)
    parser.add_argument('--fu_weight_decay', type=float, default=5e-4)

    ## iter
    parser.add_argument('--main_epoch', type=int, default=100)  # 100
    parser.add_argument('--inner_ne_epoch', type=int, default=10)
    parser.add_argument('--inner_cls_epoch', type=int, default=1)
    parser.add_argument('--inner_fu_epoch', type=int, default=1)

    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)

    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')

    parser.add_argument('--ratio', type=float, default=0.)
    parser.add_argument('--flag', type=int, default=1)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def wisconsin_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="wisconsin")

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)

    parser.add_argument('--name_view1', type=str, default="v1_knn")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_2")
    parser.add_argument('--indice_view2', type=str, default="v2_40")

    parser.add_argument('--cls_hid_1', type=int, default=16)

    ## gen
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)
    parser.add_argument('--com_lambda_v2', type=float, default=0.1)
    parser.add_argument('--gen_hid', type=int, default=64)

    ## fusion
    parser.add_argument('--fu_hid', type=float, default=16)  ###

    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.1)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.8)

    parser.add_argument('--fu_lr', type=float, default=0.01)
    parser.add_argument('--fu_weight_decay', type=float, default=5e-4)

    ## iter
    parser.add_argument('--main_epoch', type=int, default=100)  # 100
    parser.add_argument('--inner_ne_epoch', type=int, default=10)
    parser.add_argument('--inner_cls_epoch', type=int, default=1)
    parser.add_argument('--inner_fu_epoch', type=int, default=1)

    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)

    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')

    parser.add_argument('--ratio', type=float, default=0.)
    parser.add_argument('--flag', type=int, default=1)
    #####################################

    args, _ = parser.parse_known_args()
    return args


def set_params():
    if dataset == "wine":
        args = wine_params()
        args.pyg = False
        args.big = False
    elif dataset == "breast_cancer":
        args = breast_cancer_params()
        args.pyg = False
        args.big = False
    elif dataset == "digits":
        args = digits_params()
        args.pyg = False
        args.big = False
    elif dataset == "polblogs":
        args = polblogs_params()
        args.pyg = False
        args.big = False
    elif dataset == "texas":
        args = texas_params()
        args.pyg = False
        args.big = False
    elif dataset == "cornell":
        args = cornell_params()
        args.pyg = False
        args.big = False
    elif dataset == "wisconsin":
        args = wisconsin_params()
        args.pyg = False
        args.big = False
    return args
