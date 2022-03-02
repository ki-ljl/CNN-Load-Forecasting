# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/1 21:28
@Author ：KI 
@File ：cnn.py
@Motto：Hungry And Humble

"""
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from data_process import nn_seq, device, CNN_PATH, get_mape


class CNN(nn.Module):
    def __init__(self, B):
        super(CNN, self).__init__()
        self.B = B
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2),  # 30 - 2 + 1 = 29
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 29 - 2 + 1 = 28
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2),  # 28 - 2 + 1 = 27
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 27 - 2 + 1 = 26
        )
        self.Linear1 = nn.Linear(self.B * 127 * 26, self.B * 50)
        self.Linear2 = nn.Linear(self.B * 50, self.B)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.size())
        x = x.view(-1)
        # print(x.size())
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        return x


def CNN_train(file_name, B):
    Dtr, Dte, MAX, MIN = nn_seq(file_name, B)
    epochs = 10
    model = CNN(B).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print('training...')
    for epoch in range(epochs):
        cnt = 0
        for batch_idx, (seq, target) in enumerate(Dtr, 0):
            cnt = cnt + 1
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            # input size（batch size, channel, series length）
            y_pred = model(seq.reshape(B, 1, 30))
            # print('y_pred=', y_pred, 'target=', target)
            loss = loss_function(y_pred, target)
            loss.backward()
            optimizer.step()
            if cnt % 50 == 0:
                print(f'epoch: {epoch:3} loss: {loss.item():10.8f}')
        print(f'epoch: {epoch:3} loss: {loss.item():10.10f}')

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, CNN_PATH)


def CNN_predict(cnn, test_seq):
    pred = []
    y = []
    for batch_idx, (seq, target) in enumerate(test_seq, 0):
        seq = seq.to(device)
        with torch.no_grad():
            target = list(chain.from_iterable(target.tolist()))
            y.extend(target)
            pred.extend(cnn(seq.reshape(cnn.B, 1, 30)).tolist())

    y, pred = np.array([y]), np.array([pred])
    return y, pred


def test():
    file_name = 'anqiudata.csv'
    B = 15
    Dtr, Dte, MAX, MIN = nn_seq(file_name, B)
    cnn = CNN(B).to(device)
    cnn.load_state_dict(torch.load(CNN_PATH)['model'])
    cnn.eval()
    test_y, pred = CNN_predict(cnn, Dte)

    test_y = (MAX - MIN) * test_y + MIN
    pred = (MAX - MIN) * pred + MIN
    print('accuracy:', get_mape(test_y.flatten(), pred))


if __name__ == '__main__':
    file_name = 'anqiudata.csv'
    B = 15
    # CNN_train(file_name, B)
    test()

