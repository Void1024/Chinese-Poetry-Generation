#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function

from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras import optimizers
from paths import plan_model_path, plan_data_path, expander_model_path
from singleton import Singleton
from rank_words import RankedWords
from gensim import models
from plan import train_planner
import os
import numpy as np



WORD_VEC_DIM = 512
BATCH_SIZE = 64
    
class Expander(Singleton):
    def __init__(self):
        if not os.path.exists(plan_model_path):
            train_planner()
        self.model = models.Word2Vec.load(plan_model_path)
        self.net = None
        self.contexts = None
        self.targets = None
    
    def _build_net(self):
        input_tensor = Input(shape = (None, WORD_VEC_DIM))
        lstm = LSTM(512, return_sequences = True)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        lstm = LSTM(256)(dropout)
        dropout = Dropout(0.6)(lstm)
        dense = Dense(WORD_VEC_DIM, activation = 'sigmoid')(dropout)
        self.net = Model(input_tensor, dense)
        adam = optimizers.Adam(lr = 0.002, decay = 1e-6)
        self.net.compile(loss = 'mean_squared_error', optimizer = adam)
    
    def _get_train_data(self):
        self.contexts = []
        self.targets = []
        with open(plan_data_path, 'r') as fin:
            for line in fin.readlines():
                keywords = line.strip().split()
                if max([0 if word in self.model.wv else 1 for word in keywords]) > 0:
                    continue
                keywords = [self.model.wv[word] for word in keywords]
                if len(keywords) < 4:
                    continue
                context = [keywords[0]]
                for target in keywords[1:]:
                    self.contexts.append(context)
                    self.targets.append(target)
                    context.append(target)
        self.input_data = np.zeros(
            (len(self.contexts), max([len(context) for context in self.contexts]), WORD_VEC_DIM),
            dtype = 'float32'
        )
        self.output_data = np.zeros(
            (len(self.targets), WORD_VEC_DIM),
            dtype = 'float32'
        )
        for i in range(len(self.contexts)):
            for j in range(len(self.contexts[i])):
                self.input_data[i, j, :] = self.contexts[i][j]
            self.output_data[i, :] = self.targets[i]


    def train(self, epoch = 1):
        if not self.net:
            self._build_net()
        if not self.contexts:
            self._get_train_data()
        self.net.fit(
            self.input_data,
            self.output_data,
            batch_size = BATCH_SIZE,
            epochs = epoch,
            validation_split = 0.2
        )
        self.net.save(expander_model_path)

    def predict(self, keywords):
        if not os.path.exists(expander_model_path):
            self.train()
        net = load_model(expander_model_path)
        keywords = [self.model.wv[word] for word in keywords]
        input_words = np.zeros((1, len(keywords), WORD_VEC_DIM))
        for i in range(len(keywords)):
            input_words[0, i, :] = keywords[i]
        target_vec = net.predict(input_words)
        target = self.model.wv.most_similar(positive = [target_vec[0]])
        return target
        


if __name__ == '__main__':
    e = Expander()
    e.predict(['月'])


