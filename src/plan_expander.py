#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function

import tensorflow as tf
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
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session


WORD_VEC_DIM = 512
BATCH_SIZE = 64
    
class Expander(Singleton):
    def __init__(self):

        tf.keras.backend.clear_session()  # For easy reset of notebook state.

        config_proto = tf.ConfigProto()
        off = rewriter_config_pb2.RewriterConfig.OFF
        config_proto.graph_options.rewrite_options.arithmetic_optimization = off
        session = tf.Session(config=config_proto)
        set_session(session)


        if not os.path.exists(plan_model_path):
            train_planner()
        self.model = models.Word2Vec.load(plan_model_path)
        self.word_lists = []
        with open(plan_data_path, 'r') as fin:
            for line in fin.readlines():
                for word in line.strip().split('\t'):
                    if word not in self.word_lists:
                        self.word_lists.append(word)

        self.net = None
        self.contexts = None
        self.targets = None
    
    def _build_net(self):
        input_tensor = Input(shape = (None, WORD_VEC_DIM))
        lstm = LSTM(512)(input_tensor)
        dropout = Dropout(0.6)(lstm)
        # lstm = LSTM(256)(dropout)
        # dropout = Dropout(0.6)(lstm)
        dense = Dense(len(self.word_lists), activation = 'softmax')(dropout)
        # dense = Dense(len(self.word_lists), activation = 'sigmoid')(dropout)
        self.net = Model(input_tensor, dense)
        adam = optimizers.Adam(lr = 0.002, decay = 1e-6)
        self.net.compile(loss = 'categorical_crossentropy', optimizer = adam)
        # self.net.compile(loss = 'mean_squared_error', optimizer = adam)
    
    def _get_train_data(self):
        print('Generate Expander Model Train Data ...')
        self.contexts = []
        self.targets = []
        with open(plan_data_path, 'r') as fin:
            for line in fin.readlines():
                keywords = line.strip().split()
                if max([0 if word in self.model.wv else 1 for word in keywords]) > 0:
                    continue
                
                if len(keywords) < 4:
                    continue
                context = [self.model.wv[keywords[0]]]
                for target in keywords[1:]:
                    self.contexts.append(context)
                    self.targets.append(target)
                    context.append(self.model.wv[target])
        self.input_data = np.zeros(
            (len(self.contexts), max([len(context) for context in self.contexts]), WORD_VEC_DIM),
            dtype = 'float32'
        )
        self.output_data = np.zeros(
            (len(self.targets), len(self.word_lists)),
            dtype = 'float32'
        )
        for i in range(len(self.contexts)):
            for j in range(len(self.contexts[i])):
                self.input_data[i, j, :] = self.contexts[i][j]
            self.output_data[i, self.word_lists.index(self.targets[i])] = 1


    def train(self, epoch = 1):
        if os.path.exists(expander_model_path):
            self.net = load_model(expander_model_path)
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
            print('Please train Plan-Expander model first !')
            return
        net = load_model(expander_model_path)
        keywords = [self.model.wv[word] for word in keywords]
        input_words = np.zeros((1, len(keywords), WORD_VEC_DIM))
        for i in range(len(keywords)):
            input_words[0, i, :] = keywords[i]
        target_vec = net.predict(input_words)
        target = sorted(range(len(target_vec[0])), key = lambda i : target_vec[0][i])
        return self.word_lists[target[-1]], target, target_vec
        


if __name__ == '__main__':
    e = Expander()
    e.train(10)


