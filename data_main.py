#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import numpy as np
from utils import data_masks, graph_constrct


class Main_Train_Data():
    def __init__(self, train_seqs, opt):
        inputs, targets = [], []
        for seq in train_seqs:
            inputs.append(seq[:-1])
            targets.append(seq[-1])

        inputs, mask, len_max = data_masks(inputs, [0])

        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(targets)
        self.length = len(inputs)
        self.opt = opt

    def generate_batch(self, batch_size, shuffled_arg):
        self.inputs = self.inputs[shuffled_arg]
        self.mask = self.mask[shuffled_arg]
        self.targets = self.targets[shuffled_arg]

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        alias_inputs, A, items = graph_constrct(inputs)
        return alias_inputs, A, items, mask, targets

class Main_Test_Data():
    def __init__(self, test_seqs, opt):
        inputs, targets = [], []
        for seq in test_seqs:
            for i in range(1, len(seq)):
                inputs.append(seq[:i])
                targets.append(seq[i])

        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(targets)
        self.length = len(inputs)
        self.opt = opt

    def generate_batch(self, batch_size):
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        alias_inputs, A, items = graph_constrct(inputs)
        return alias_inputs, A, items, mask, targets
