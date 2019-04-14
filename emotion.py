#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char_dict import CharDict
from gensim import models
from numpy.random import uniform
from paths import char2vec_path, check_uptodate
from poems import Poems
from singleton import Singleton
from utils import CHAR_VEC_DIM
from paths import emotion_dir, emotion_poems_path, check_uptodate
import numpy as np
import os

_emotion_word2vec_model_path = os.path.join(emotion_dir, 'emotion_vec_model.bin')

emotion_list = ['悲', '惧', '乐', '怒', '思', '喜', '忧']
emotion_dict = {
    '悲':['悲','愁','恸','痛','寡','哀','伤','嗟','凄','怨','苦'],
    '惧':['惧','谗','谤','患','罪','诈','诬','惮','忌'],
    '乐':['乐','悦','欣','怡','洽','畅','愉','欢','洽'],
    '怒':['怒','雷','吼','霆','霹','猛','轰','震'],
    '思':['思','忆','怀','恨','吟','逢','期','乡','旅','送'],
    '喜':['喜','健','倩','贺','好','良','善','幸','赏','朋'],
    '忧':['忧','恤','痾','虑','艰','遑','厄','迫'],
}

class PriorEmotion(Singleton):
    def __init__(self):
        self.char_dict = CharDict()
        poems = Poems()
        self.poems = [' '.join(poetry) for poetry in poems]
        if not os.path.exists(_emotion_word2vec_model_path):
            if not os.path.exists(emotion_dir):
                os.mkdir(emotion_dir)
            self.model = models.Word2Vec(self.poems, size = CHAR_VEC_DIM)
            self.model.save(_emotion_word2vec_model_path)
        else:
            self.model = models.Word2Vec.load(_emotion_word2vec_model_path)
    def predict(self, poetry):
        emotions = [0.0 for emotion in emotion_list]
        sentences = poetry.split()

        for idx in range(len(sentences)):
            sentence = sentences[idx]
            for ch in sentence:
                for i in range(len(emotions)):
                    if ch in emotion_dict[emotion_list[i]]:
                        emotions[i] += 1.0
                    elif ch in self.model.wv:
                        emotions[i] += self.model.wv.similarity(ch, emotion_list[i])

        max_prob = sorted(range(len(emotions)), key = lambda k : emotions[k])[-1]
        predict = emotion_list[max_prob]
        
        return predict
    def gen_train_data(self):
        if not os.path.exists(emotion_dir):
            os.mkdir(emotion_dir)
        with open(emotion_poems_path, 'w', encoding = 'utf-8') as fout:
            for poetry in self.poems:
                sentences_len = list(map(lambda p : len(p), poetry.split()))
                if len(sentences_len) != 8 and len(sentences_len) != 4:
                    continue
                if sentences_len[0] < 5:
                    continue
                emotion = self.predict(poetry)
                fout.write(' '.join((emotion, poetry)) + '\n')
                



if __name__ == '__main__':
    pass