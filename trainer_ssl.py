#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import torch
from utils import trans_to_cuda, self_super_pair_mask


def pretrain_user2user_curuser(model, user2user_curuser):
    cur_alias_inputs, cur_A, cur_items, cur_mask = user2user_curuser
    cur_alias_inputs = trans_to_cuda(torch.Tensor(cur_alias_inputs).long())
    cur_items = trans_to_cuda(torch.Tensor(cur_items).long())
    cur_A = trans_to_cuda(torch.Tensor(cur_A).float())
    cur_mask = trans_to_cuda(torch.Tensor(cur_mask).long())
    # propagation
    hidden = model.propagation(cur_items, cur_A)
    # graph to seq
    get = lambda i: hidden[i][cur_alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(cur_alias_inputs)).long()])
    # aggregation
    a = model.long_term(seq_hidden, cur_mask)
    return a

def pretrain_user2user_posneg_user(model, user2user_users):
    alias_inputs, A, items, mask = user2user_users
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())

    bs, user_neighbor_num = items.shape[0], items.shape[1]

    # reshape to maintain the requirements of gnn
    alias_inputs = alias_inputs.view(-1, alias_inputs.shape[2])
    items = items.view(-1, items.shape[2])
    A = A.view(-1, A.shape[2], A.shape[3])
    mask = mask.view(-1, mask.shape[2])

    # propagation
    hidden = model.propagation(items, A)
    # graph to seq
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # aggregation
    a = model.long_term(seq_hidden, mask)

    neighbor_mask = torch.sign(torch.sum(mask, -1))

    neighbor_mask = neighbor_mask.view(bs, user_neighbor_num)
    a = a.view(bs, user_neighbor_num, a.shape[-1])
    return a, neighbor_mask

def pretrain_user2user(model, user2user_curuser, user2user_posusers, user2user_negusers, opt):
    cur_user_pref = pretrain_user2user_curuser(model, user2user_curuser)
    pos_users_pref, neighbor_mask = pretrain_user2user_posneg_user(model, user2user_posusers)
    neg_users_pref, _ = pretrain_user2user_posneg_user(model, user2user_negusers)

    cur_user_pref = model.LayerNorm(cur_user_pref)
    pos_users_pref = model.LayerNorm(pos_users_pref)
    neg_users_pref = model.LayerNorm(neg_users_pref)

    pos_socres = torch.sum(cur_user_pref.unsqueeze(1).repeat(1, pos_users_pref.shape[1], 1) * pos_users_pref, -1)
    neg_socres = torch.sum(cur_user_pref.unsqueeze(1).repeat(1, neg_users_pref.shape[1], 1) * neg_users_pref, -1)

    loss = self_super_pair_mask(pos_socres, neg_socres, neighbor_mask)
    return loss

def user_train_forward(model, i, data, opt):
    user2user_curuser, user2user_posusers, user2user_negusers = data.get_slice(i)
    user2user_loss = pretrain_user2user(model, user2user_curuser, user2user_posusers, user2user_negusers, opt)
    return user2user_loss
