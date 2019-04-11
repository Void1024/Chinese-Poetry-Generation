#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char_dict import CharDict
from gensim import models
from numpy.random import uniform
from paths import char2vec_path, check_uptodate
from poems import Poems
from singleton import Singleton
from utils import CHAR_VEC_DIM
import numpy as np
import os



emotion_list = ['喜', '怒', '乐', '悲', '惧', '思', '忧']

class EmotionDict(Singleton):
    def __init__(self):
        self.char_dict = CharDict()
        poems = Poems()
        self.poems = [' '.join(poetry) for poetry in poems]
        self.model = models.Word2Vec(self.poems, size = CHAR_VEC_DIM)
    def predict(self, sentences):
        emotions = [0.0 for emotion in emotion_list]
        for sentence in sentences:
            for ch in sentence:
                for i in range(len(emotions)):
                    emotions[i] += self.model.similarity(ch, emotion_list[i])
        max_prob = sorted(range(len(emotions)), key = lambda k : emotions[k])[-1]
        predict = emotion_list[max_prob]
        
        return predict


if __name__ == '__main__':
    d = EmotionDict()
    for i, ch in enumerate(d.char_dict):
        if ch in d.model.wv:
            # print(ch)
            # print(d.model.wv[ch])
            print('in')
        print(ch)