# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/1 21:28
@Author ：KI 
@File ：cnn.py
@Motto：Hungry And Humble

"""
import copy
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


def get_val_loss(model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq.reshape(B, 1, 30))
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train():
    model = CNN(B).to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print('training...')
    epochs = 30
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in range(epochs):
        train_loss = []
        for batch_idx, (seq, target) in enumerate(Dtr, 0):
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            # input size（batch size, channel, series length）
            y_pred = model(seq.reshape(B, 1, 30))
            # print('y_pred=', y_pred, 'target=', target)
            loss = loss_function(y_pred, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation
        val_loss = get_val_loss(model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'model': best_model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, CNN_PATH)


def predict(cnn, test_seq):
    pred = []
    y = []
    for batch_idx, (seq, target) in enumerate(test_seq, 0):
        seq = seq.to(device)
        with torch.no_grad():
            target = list(chain.from_iterable(target.tolist()))
            y.extend(target)
            pred.extend(cnn(seq.reshape(cnn.B, 1, 30)).tolist())

    y, pred = np.array(y), np.array(pred)
    return y, pred


def test():
    cnn = CNN(B).to(device)
    cnn.load_state_dict(torch.load(CNN_PATH)['model'])
    cnn.eval()
    test_y, pred = predict(cnn, Dte)

    test_y = (MAX - MIN) * test_y + MIN
    pred = (MAX - MIN) * pred + MIN
    print('mape:', get_mape(test_y, pred))


if __name__ == '__main__':
    file_name = 'data.csv'
    B = 15
    Dtr, Val, Dte, MAX, MIN = nn_seq(file_name, B)
    train()
    test()

