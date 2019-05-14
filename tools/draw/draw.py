#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt

log_path = 'log/log.txt'
def get_data(path):
    with open(path, 'r', encoding = 'utf-8') as fin:
        train_loss = []
        val_loss = []
        epoch_list = []
        epoch = 0
        for line in fin.readlines():
            data_list = line.strip().split()
            if data_list[0] == 'train':
                train_loss.append(float(data_list[2]))
                epoch += 1
                epoch_list.append(epoch)
            elif data_list[0] == 'validate':
                val_loss.append(float(data_list[2]))
            else:
                continue
    return train_loss, val_loss, epoch_list

def draw(path):
    train_loss, val_loss, epoch_list = get_data(path)

    plt.plot(epoch_list, train_loss, color = 'green', label = 'train loss')
    plt.plot(epoch_list, val_loss, color = 'blue', label = 'validate loss')

    plt.savefig('tmp.jpg')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

            
