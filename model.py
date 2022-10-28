#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import math
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class LayerNorm(Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = Parameter(torch.ones(hidden_size))
        self.bias = Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)

        # long-term
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        # hybrid
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_six = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.alpha = Parameter(torch.Tensor(1))

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def propagation(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden

    def long_term(self, hidden, mask):
        mask_sum = torch.sum(mask, -1, keepdim=True) # bs * seq_length * 1
        mask_sum[torch.where(mask_sum==0)] = 1 # bs * seq_length * 1
        ht = torch.sum(hidden*mask.unsqueeze(-1), 1) / mask_sum # bs * seq_length * latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))

        alpha = torch.exp(alpha.squeeze(-1))
        alpha = alpha * mask.view(mask.shape[0], -1).float()
        alpha_sum = torch.sum(alpha, -1, keepdim=True)
        alpha_sum[torch.where(alpha_sum==0)] = 1
        alpha = alpha / alpha_sum
        alpha = alpha.unsqueeze(-1)

        a = torch.sum(alpha * hidden, 1)
        return a

    def hybrid(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_four(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_five(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_six(torch.sigmoid(q1 + q2))

        alpha = torch.exp(alpha.squeeze(-1))
        alpha = alpha * mask.view(mask.shape[0], -1).float()
        alpha_sum = torch.sum(alpha, -1, keepdim=True)
        alpha_sum[torch.where(alpha_sum==0)] = 1
        alpha = alpha / alpha_sum
        alpha = alpha.unsqueeze(-1)
        
        a = torch.sum(alpha * hidden, 1)
        a = self.linear_transform(torch.cat([a, ht], 1))
        return a

    def compute_scores(self, a):
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores
