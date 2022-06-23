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
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from data_process import nn_seq, device, CNN_PATH, get_mape

B, input_size = 15, 7


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.B = B
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=2),  # 24 - 2 + 1 = 23
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),  # 23 - 2 + 1 = 22
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2),  # 22 - 2 + 1 = 21
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),  # 21 - 2 + 1 = 20
        )
        self.Linear1 = nn.Linear(self.B * 127 * 20, self.B * 50)
        self.Linear2 = nn.Linear(self.B * 50, self.B)

    def forward(self, x):
        # (batch size, channel, series length)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1)
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)

        x = x.view(x.shape[0], -1)

        return x


def get_val_loss(model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)


def train(Dtr, Val):
    model = CNN().to(device)
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
            y_pred = model(seq)
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


def test(Dte, m, n):
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load(CNN_PATH)['model'])
    cnn.eval()
    pred = []
    y = []
    for batch_idx, (seq, target) in enumerate(Dte, 0):
        seq = seq.to(device)
        with torch.no_grad():
            target = list(chain.from_iterable(target.tolist()))
            y.extend(target)
            pred.extend(cnn(seq).tolist())

    y, pred = np.array(y), np.array(pred)

    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    x = [i for i in range(1, 151)]
    x_smooth = np.linspace(np.min(x), np.max(x), 900)
    y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')

    y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()


def main():
    file_name = 'data.csv'
    Dtr, Val, Dte, MAX, MIN = nn_seq(file_name, B)
    train(Dtr, Val)
    test(Dte, MAX, MIN)


if __name__ == '__main__':
    main()

