#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import torch
import datetime
import numpy as np
from trainer_ssl import user_train_forward
from utils import trans_to_cuda, trans_to_cpu, ce_lcl


def main_train(model, train_data, user_data, opt):
    
    print('start training: ', datetime.datetime.now())
    print('learning rate = ', model.optimizer.state_dict()['param_groups'][0]['lr'])
    model.train()
    total_loss = 0.0
    shuffled_arg = np.arange(train_data.length)
    np.random.shuffle(shuffled_arg)
    slices = train_data.generate_batch(model.batch_size, shuffled_arg)
    user_data.generate_batch(shuffled_arg)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        
        # main loss
        loss = main_train_forward(model, i, train_data, opt)

        # user loss
        user_loss = user_train_forward(model, i, user_data, opt)
        loss = loss + opt.ssl_lambda*user_loss
        
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

def main_test(model, test_data, opt):
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = main_test_forward(model, i, test_data, opt)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, targets):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr

def prediction(model, test_data, opt, output_folder, test_type):
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    ranks = []
    for i in slices:
        targets, scores = main_test_forward(model, i, test_data, opt)
        sub_scores = scores.topk(20)[1]
        all_scores = scores.topk(opt.n_node-1)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        all_scores = trans_to_cpu(all_scores).detach().numpy()
        for score, all_score, target in zip(sub_scores, all_scores, targets):
            rank = np.where(all_score == target - 1)[0][0] + 1
            ranks.append(rank)
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100

    rank_write = open('output/'+output_folder+'/'+test_type+'_rank.txt', 'w')
    for rank in ranks:
        rank_write.write(str(rank) + '\n')
    rank_write.flush()
    rank_write.close()
    return hit, mrr

def main_train_test(model, train_data, user_data, test_data, opt):
    main_train(model, train_data, user_data, opt)
    hit, mrr = main_test(model, test_data, opt)
    return hit, mrr

def main_train_forward(model, i, data, opt):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    targets = trans_to_cuda(torch.Tensor(targets).long())
    # propagation
    hidden = model.propagation(items, A)
    # graph to seq
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # aggregation
    a = model.hybrid(seq_hidden, mask)

    loss = ce_lcl(model, a, targets - 1, opt)
    return loss

def main_test_forward(model, i, data, opt):
    alias_inputs, A, items, inp_mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    inp_mask = trans_to_cuda(torch.Tensor(inp_mask).long())
    # propagation
    hidden = model.propagation(items, A)
    # graph to seq
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    # aggregation
    a = model.hybrid(seq_hidden, inp_mask)
    return targets, model.compute_scores(a)
