# -*- coding:utf-8 -*-
"""
@Time: 2022/03/01 23:16
@Author: KI
@File: data_process.py
@Motto: Hungry And Humble
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_PATH = 'model/CNN.pkl'


def load_data(file_name):
    df = pd.read_csv('data/' + file_name, encoding='gbk')
    df.fillna(df.mean(), inplace=True)

    return df


def nn_seq(file_name, B):
    print('data processing...')
    dataset = load_data(file_name)
    # split
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data):
        load = data[data.columns[1]]
        data = data.values.tolist()
        load = (load - n) / (m - n)
        load = load.tolist()
        seq = []
        for i in range(len(data) - 30):
            train_seq = []
            train_label = []
            for j in range(i, i + 24):
                train_seq.append(load[j])
            for c in range(2, 8):
                train_seq.append(data[i + 24][c])
            train_label.append(load[i + 24])
            train_seq = torch.FloatTensor(train_seq).view(-1)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)

        seq = DataLoader(dataset=seq, batch_size=B, shuffle=True, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train)
    Val = process(val)
    Dte = process(test)

    return Dtr, Val, Dte, m, n


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def get_mape(x, y):
    """
    :param x:true
    :param y:pred
    :return:MAPE
    """
    return np.mean(np.abs((x - y) / x))
