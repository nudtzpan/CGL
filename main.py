#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
import pickle
import time
from utils import trans_to_cuda
from trainer_main import main_train_test, prediction
from model import SessionGraph
import os
import numpy as np
import torch
import random
from data_main import Main_Train_Data, Main_Test_Data
from data_ssl import User_Data


def init_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: Retailrocket/Diginetica/Gowalla/sample')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=0, help='l2 penalty')
parser.add_argument('--epoch_num', type=int, default=10, help='epoch number')
parser.add_argument('--patience', type=int, default=2, help='early stop')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')

parser.add_argument('--n_node', type=int, default=0, help='the number of the items in the dataset')
parser.add_argument('--gpu', type=str, default='0', help='')

parser.add_argument('--ssl_lambda', type=float, default=0.1, help='') # [0.05, 0.1, 0.5, 1]
parser.add_argument('--smooth', type=float, default=0.1, help='for label smooth')
parser.add_argument('--alpha', type=float, default=10, help='for label confusion')

opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

output_folder = opt.dataset+str(np.random.randint(0, 1e5))

f = open('./datasets/' + opt.dataset + '/' + 'detail.txt','r')
opt.n_node = int(f.readlines()[0].strip())+1
print(opt)

init_seed(2021)
def main():
    
    train_seqs = pickle.load(open('./datasets/' + opt.dataset + '/aug_train_seqs.txt', 'rb'))
    valid_seqs = pickle.load(open('./datasets/' + opt.dataset + '/valid_seqs.txt', 'rb'))
    test_seqs = pickle.load(open('./datasets/' + opt.dataset + '/test_seqs.txt', 'rb'))
    user_users = pickle.load(open('./datasets/' + opt.dataset+'/user_users.txt','rb'))
    user_items = pickle.load(open('./datasets/' + opt.dataset + '/user_items.txt','rb'))

    # for main componnet
    train_data = Main_Train_Data(train_seqs, opt)
    valid_data = Main_Test_Data(valid_seqs, opt)
    test_data = Main_Test_Data(test_seqs, opt)
    print ('train_data.length = ', np.sum(train_data.length))
    print ('valid_data.length = ', valid_data.length)
    print ('test_data.length = ', test_data.length)
    # for user side component
    user_data = User_Data(train_seqs, user_users, user_items, opt=opt)

    model = trans_to_cuda(SessionGraph(opt, opt.n_node))

    if not os.path.exists('output/'):
        os.mkdir('output/')
    if not os.path.exists('output/'+output_folder):
        os.mkdir('output/'+output_folder)

    with open('output/'+output_folder+'/paras.txt', 'a') as f:
        f.write(str(opt) + '\n')

    start = time.time()
    best_result = [-1, -1]
    best_epoch = [-1, -1]
    bad_counter = 0
    for epoch in range(opt.epoch_num):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = main_train_test(model, train_data, user_data, valid_data, opt)

        if hit > best_result[0] or mrr > best_result[1]:
            bad_counter = 0
        else:
            bad_counter += 1

        if hit > best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            torch.save(model, 'output/'+output_folder+'/recall.pkl')

        if mrr > best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            torch.save(model, 'output/'+output_folder+'/mrr.pkl')

        print('Valid Best Result:')
        print('Valid\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')

    # for test
    model = torch.load('output/'+output_folder+'/recall.pkl')
    test_hit, _ = prediction(model, test_data, opt, output_folder, 'recall')
    model = torch.load('output/'+output_folder+'/mrr.pkl')
    _, test_mrr = prediction(model, test_data, opt, output_folder, 'mrr')
    print('Test\tRecall@20:\t%.4f\tMMR@20:\t%.4f'% (test_hit, test_mrr))

    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()
