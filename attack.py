import torch
import numpy as np
import torch.nn.functional as F
# from attack_model.random_attack import random_attack
# from attack_model.mettack import meta_attack
# from attack_model.dice import dice_attack

from attack_model.mettack import Metattack
from attack_model.random_attack import Random
from attack_model.dice import DICE
from scipy import sparse
import copy
import torch_geometric as pyg

def meta_attack(args, trained_model, data, perturbation_num):
    adj = pyg.utils.to_dense_adj(data.edge_index).squeeze()
    adj = sparse.csr_matrix(adj.cpu().numpy())
    features = sparse.csr_matrix(data.x.cpu().numpy())
    labels = data.y.cpu().numpy()

    surrogate = copy.deepcopy(trained_model)
    model = Metattack(args=args, model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=args.device)

    idx_train = (data.train_mask == True).nonzero(as_tuple=True)[0].cpu().numpy()
    idx_val = (data.val_mask == True).nonzero(as_tuple=True)[0].cpu().numpy()
    idx_test = (data.test_mask == True).nonzero(as_tuple=True)[0].cpu().numpy()
    idx_unlabeled = np.union1d(idx_val, idx_test)
    print()


    model.attack(data, features, adj, labels, idx_train, idx_unlabeled, perturbation_num)
    modified_adj = model.modified_adj
    # np.save('result/adj/{}/meta_adj.npy'.format(args.dataset), modified_adj.detach().cpu().numpy())
    # np.save('result/Meta_acm/meta_acc_{}.npy'.format(index), np.array(model.acc_list))

    # output = pa_predict(model, features, modified_adj)
    # features2 = torch.FloatTensor(features.todense()).to(device)
    # output2 = surrogate.predict(features2, modified_adj)
    #
    # acc_pa = accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()
    # acc_ea = accuracy(output2[idx_unlabeled], labels[idx_unlabeled]).item()
    # print(acc_pa)
    # print(acc_ea)
    # print('This is Meta-both')
    # asr_pa = ASR_PA(model, features, features, labels, idx_test, adj, modified_adj, device)
    # asr_ea = ASR_EA(surrogate, features, features, labels, idx_test, adj, modified_adj, device)
    # print('done')
    return modified_adj


def random_attack(data, perturbation_num):
    adj = pyg.utils.to_dense_adj(data.edge_index).squeeze()
    adj = sparse.csr_matrix(adj.cpu().numpy())
    model = Random()
    model.modified_adj = adj
    model.attack(model.modified_adj, n_perturbations=perturbation_num, type='flip')
    modified_adj = model.modified_adj
    modified_adj = torch.HalfTensor(modified_adj.todense())
    return modified_adj


def dice_attack(data, perturbation_num):
    adj = pyg.utils.to_dense_adj(data.edge_index).squeeze()
    adj = sparse.csr_matrix(adj.cpu().numpy())
    model = DICE()
    model.modified_adj = adj
    model.attack(model.modified_adj, data.y, n_perturbations=perturbation_num)
    modified_adj = model.modified_adj
    modified_adj = torch.HalfTensor(modified_adj.todense())
    return modified_adj

