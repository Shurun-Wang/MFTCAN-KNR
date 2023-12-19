#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class MFP2TCANet(nn.Module):
    """
    main structure
    """
    def __init__(self):
        super(MFP2TCANet, self).__init__()
        self.feature1 = TCANet(emb_size=1, num_channels=[25, 25, 25], key_size=25, kernel_size=15, dropout=0.5)
        self.feature2 = TCANet(emb_size=1, num_channels=[25, 25, 25], key_size=25, kernel_size=15, dropout=0.5)
        self.linear = nn.Linear(2, 1)
        # self.linear2 = nn.Linear(10, 1)
        self.LeakyReLU = nn.LeakyReLU()


    def forward(self, feature1, feature2):
        f1 = self.feature1(feature1)
        f2 = self.feature2(feature2)
        cat = torch.cat((f1, f2), dim=1)
        out = self.linear(cat)
        out = self.LeakyReLU(out)
        return cat, out

class MFPTCANet(nn.Module):
    """
    main structure
    """
    def __init__(self):
        super(MFPTCANet, self).__init__()
        self.feature1 = TCANet(emb_size=1, num_channels=[25, 25, 25], key_size=25, kernel_size=15, dropout=0.5)
        self.linear = nn.Linear(25, 1)
        # self.linear2 = nn.Linear(10, 1)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, feature1, feature2):
        f1 = self.feature1(feature1)
        out = self.linear(f1[:, :, -1])
        out = self.LeakyReLU(out)
        return f1[:, :, -1], out


class TCANet(nn.Module):
    """
    main structure
    """
    def __init__(self, emb_size, num_channels, key_size, kernel_size, dropout):
        super(TCANet, self).__init__()
        self.tcanet = TemporalConvNet(emb_size, num_channels, key_size, kernel_size, dropout)
        self.linear = nn.Linear(25, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        y = self.tcanet(input) # input should have dimension (N, C, L)
        o = self.linear(y[:, :, -1])
        o = self.relu(o)
        return o

class TemporalConvNet(nn.Module):
    """
    second main structure
    """
    def __init__(self, emb_size, num_channels, key_size, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = emb_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, key_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batchsize, seq_len, emb_size]
        return self.network(x)


class TemporalBlock(nn.Module):
    """
    Block structure
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, key_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.attention = AttentionBlock(n_inputs, key_size, n_inputs)
        # n_inputs = 1, n_outputs = 25
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.LeakyReLU = nn.LeakyReLU()
        self.net = self._make_layers(n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout)

    def _make_layers(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        layers_list = []
        # n_inputs = 1, n_outputs = 25, kernel_size = 7, stride = 1
        layers_list.append(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        layers_list.append(nn.BatchNorm1d(n_outputs))
        layers_list.append(Chomp1d(padding))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(dropout))
        layers_list.append(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        layers_list.append(nn.BatchNorm1d(n_outputs))
        layers_list.append(Chomp1d(padding))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(dropout))
        return nn.Sequential(*layers_list)


    def forward(self, x):
        # x: [N, emb_size, T]
        # x: [64, 1, 784]
        out_attn, attn_weight = self.attention(x)
        out = self.net(out_attn)
        # attn_weight: [64, 784, 784]
        weight_x = F.softmax(attn_weight.sum(dim=2), dim=1)
        # weight_x: [64, 784], x :[64, 1, 784]
        en_res_x = weight_x.unsqueeze(2).repeat(1, 1, x.size(1)).transpose(1, 2) * x
        en_res_x = en_res_x if self.downsample is None else self.downsample(en_res_x)
        res = x if self.downsample is None else self.downsample(x)
        return self.LeakyReLU(out + res + en_res_x)


class AttentionBlock(nn.Module):
    """
    output the attention result and the attention weight map
    """
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
        self.cuda = torch.device('cuda:0')

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        # mask: low triangle are zeros, later we will fill them with values
        # mask = np.array([[1 if i > j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        # mask = torch.tensor(mask, dtype=torch.uint8).to(self.cuda)
        mask = torch.tensor([[1 if i > j else 0 for i in range(input.size(2))] for j in range(input.size(2))])
        mask = torch.tensor(mask, dtype=torch.uint8).to(self.cuda)
        # mask = torch.ByteTensor(mask).to(torch.device('cuda:0'))
        input = input.permute(0, 2, 1)  # input: [N, T, inchannels] [64, 784, 1]
        keys = self.linear_keys(input)  # keys: (N, T, key_size) [64, 784, 25]
        query = self.linear_query(input)  # query: (N, T, key_size) [64, 784, 25]
        values = self.linear_values(input)  # values: (N, T, value_size) [64, 784, 1]
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))#.to(self.cuda) # shape: (N, T, T) [64, 784, 1] 涓夌淮鐭╅樀鐨勪箻娉?
        temp.data.masked_fill_(mask, -float('inf'))
        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1)
        value_attentioned = torch.bmm(weight_temp, values).permute(0, 2, 1)#.to(self.cuda)  # shape: (N, T, value_size)
        return value_attentioned, weight_temp  # value_attentioned: [N, in_channels, T], weight_temp: [N, T, T]


class Chomp1d(nn.Module):
    """
    just a tool for cutting the extra padding
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


if __name__ == '__main__':
    pass