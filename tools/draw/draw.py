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

def draw_img():
    train_loss1, val_loss1, epoch_list = get_data('log/log1.txt')
    train_loss2, val_loss2, epoch_list = get_data('log/log2.txt')
    train_loss3, val_loss3, epoch_list = get_data('log/log3.txt')
    train_loss4, val_loss4, epoch_list = get_data('log/log4.txt')
    train_loss5, val_loss5, epoch_list = get_data('log/log5.txt')
    train_loss6, val_loss6, epoch_list = get_data('log/log6.txt')

    plt.plot(epoch_list, train_loss1, color = 'green', label = 'test1 train loss')
    plt.plot(epoch_list, train_loss2, color = 'red', label = 'test2 train loss')
    plt.plot(epoch_list, train_loss3, color = 'blue', label = 'test3 train loss')
    plt.plot(epoch_list, train_loss4, color = 'pink', label = 'test4 train loss')
    plt.plot(epoch_list, train_loss5, color = 'yellow', label = 'test5 train loss')
    plt.plot(epoch_list, train_loss6, color = 'black', label = 'test6 train loss')
    # plt.plot(epoch_list, val_loss, color = 'blue', label = 'validate loss')

    plt.savefig('tmp.jpg')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

            
if __name__ == '__main__':
    draw_img()