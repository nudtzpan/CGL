#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max

def neg_sample(user_set, user_size):
    item = random.sample(range(1, user_size+1), 1)[0]
    while item in user_set:
        item = random.sample(range(1, user_size+1), 1)[0]
    return item

def graph_constrct(inputs):
    items, n_node, A, alias_inputs = [], [], [], []
    for u_input in inputs:
        n_node.append(len(np.unique(u_input)))
    max_n_node = np.max(n_node)
    for u_input in inputs:
        node = np.unique(u_input)
        items.append(node.tolist() + (max_n_node - len(node)) * [0])
        u_A = np.zeros((max_n_node, max_n_node))
        for i in np.arange(len(u_input) - 1):
            if u_input[i + 1] == 0:
                break
            u = np.where(node == u_input[i])[0][0]
            v = np.where(node == u_input[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        A.append(u_A)
        alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
    return alias_inputs, A, items

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def ce_lcl(model, a, y, opt):
    x = model.compute_scores(a)

    fake_bs = x.shape[0]
    x = torch.softmax(x, -1)

    tar_meb = model.embedding(y+1)
    items_emb = model.embedding.weight[1:]
    label_fake = torch.matmul(tar_meb, items_emb.transpose(-2, -1))
    label_fake = torch.softmax(label_fake, -1)

    label_true = torch.zeros_like(label_fake)
    label_true[torch.arange(fake_bs).long(), y] = 1
    label = opt.alpha*label_true + label_fake
    label = torch.softmax(label, -1)

    loss_temp = label * (torch.log(label+1e-16)-torch.log(x+1e-16))
    loss_temp = torch.sum(loss_temp, -1)
    loss = torch.mean(loss_temp)
    return loss

def self_super_pair_mask(pos_scores, neg_scores, mask):
    loss = torch.log(torch.sigmoid(pos_scores)+1e-16) + torch.log(1-torch.sigmoid(neg_scores)+1e-16)
    loss = torch.sum(loss*mask, -1)
    mask_sum = torch.sum(mask, -1)
    batch_mask = torch.sign(mask_sum)

    mask_sum[torch.where(mask_sum==0)] = 1
    loss = loss/mask_sum

    loss = -torch.sum(loss*batch_mask) / (torch.sum(batch_mask)+1e-16)
    return loss
