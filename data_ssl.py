#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import numpy as np
from utils import data_masks, neg_sample, graph_constrct


class User_Data():
    def __init__(self, train_seqs, user_users, user_items, opt=None):
        user_idx = 0
        user_num = len(user_users)
        
        for user in user_users:
            neighs = user_users[user]
            neighs = neighs[:1]
            user_users[user] = neighs
        
        cur_user, pos_users, neg_users = [], [], []
        for sess in train_seqs:
            user_idx += 1
            this_pos_users = user_users[user_idx]
            # the current user
            cur_user.append(user_idx)
            # pos neighbors
            pos_users.append(this_pos_users)
            # neg neighbors
            this_neg_users = []
            for i in range(len(this_pos_users)):
                this_neg_users.append(neg_sample(this_pos_users, user_num))
            neg_users.append(this_neg_users)

        user_items_list = []
        for user_idx in user_items:
            user_items_list.append(user_items[user_idx])

        pos_users, pos_mask, pos_len_max = data_masks(pos_users, [0])
        neg_users, neg_mask, neg_len_max = data_masks(neg_users, [0])
        # padding user_items for user to index the items
        user_items, user_items_mask, user_items_len_max = data_masks(user_items_list, [0])
        user_items.insert(0, [0]*user_items_len_max)
        user_items_mask.insert(0, [0]*user_items_len_max)

        # user2user cur user
        self.cur_user = np.asarray(cur_user)
        # user2user pos users
        self.pos_users = np.asarray(pos_users)
        self.pos_mask = np.asarray(pos_mask)
        # user2user neg users
        self.neg_users = np.asarray(neg_users)
        self.neg_mask = np.asarray(neg_mask)

        # user2items
        self.user_items = np.asarray(user_items)
        self.user_items_mask = np.asarray(user_items_mask)

        self.opt = opt

    def generate_batch(self, shuffled_arg):
        self.cur_user = self.cur_user[shuffled_arg]
        self.pos_users = self.pos_users[shuffled_arg]
        self.pos_mask = self.pos_mask[shuffled_arg]
        self.neg_users = self.neg_users[shuffled_arg]
        self.neg_mask = self.neg_mask[shuffled_arg]

    def user2user_curuser_slice(self, i):
        cur_user = self.cur_user[i]
        
        # construct graph for cur user
        cur_inputs, cur_mask = self.user_items[cur_user], self.user_items_mask[cur_user]
        cur_alias_inputs, cur_A, cur_items = graph_constrct(cur_inputs)
        return cur_alias_inputs, cur_A, cur_items, cur_mask

    def user2user_posneg_users_slice(self, users):        
        # construct graph for users
        inputs, mask = self.user_items[users], self.user_items_mask[users]
        bs, user_neighbors_num, user_items_num = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        inputs = np.reshape(inputs, (-1, user_items_num))
        alias_inputs, A, items = graph_constrct(inputs)

        node_num = [len(np.unique(input)) for input in inputs]
        max_node_num = max(node_num)

        alias_inputs = np.reshape(np.array(alias_inputs), (bs, user_neighbors_num, user_items_num)).tolist()
        A = np.reshape(np.array(A), (bs, user_neighbors_num, max_node_num, 2*max_node_num)).tolist()
        items = np.reshape(np.array(items), (bs, user_neighbors_num, max_node_num)).tolist()
        return alias_inputs, A, items, mask

    def get_slice(self, i):
        # user2user curuser
        user2user_curuser = self.user2user_curuser_slice(i)
        # user2user pos users
        pos_users = self.pos_users[i]
        user2user_posusers = self.user2user_posneg_users_slice(pos_users)
        # user2user neg users
        neg_users = self.neg_users[i]
        user2user_negusers = self.user2user_posneg_users_slice(neg_users)
        return user2user_curuser, user2user_posusers, user2user_negusers
