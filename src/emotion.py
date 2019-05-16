#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char_dict import CharDict
from gensim import models
from numpy.random import uniform
from paths import char2vec_path, check_uptodate
from poems import Poems, _gen_poems
from singleton import Singleton
from utils import CHAR_VEC_DIM
from paths import emotion_dir, emotion_poem_corpus, sentiment_dict_path, sentiment_word_dict_path, check_uptodate
from segment import Segmenter
from w2v import get_model
import numpy as np
import os

_corpus_list = ['qts_tab.txt', 'qsc_tab.txt', 'qss_tab.txt', 'qtais_tab.txt']

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



def _tf_idf(poems):
    word_dict = {}
    total_words = 0
    total_poem = len(poems)
    for poem in poems:
        total_words += len(poem)
        for word in poem:  
            if word not in word_dict:
                word_dict[word] = [0, 0]
            else:
                word_dict[word][0] += 1
        for word in set(poem):
            word_dict[word][1] += 1
    for word in word_dict:
        word_dict[word][0] /= total_words
        word_dict[word][1] = total_poem / (word_dict[word][1] + 1)
        word_dict[word] = word_dict[word][0] * word_dict[word][1]
    return word_dict

def _build_sentiment_dict():
    _gen_poems(_corpus_list, emotion_poem_corpus)
    poems = Poems(emotion_poem_corpus)
    poems = [' '.join(poem) for poem in poems]
    model = models.Word2Vec(poems, size = 200)
    poems = Poems()
    poems = [list(''.join(poem)) for poem in poems]
    char2score = _tf_idf(poems)
    sentiment_dict = {}
    for char in char2score:
        if char not in model.wv:
            continue
        sentiment_dict[char] = []
        for emotion in emotion_list:
            score = 0.0
            for seed_word in emotion_dict[emotion]:
                score += char2score[char] * model.wv.similarity(char, seed_word)
            sentiment_dict[char].append(score / len(emotion_dict[emotion]))
    with open(sentiment_dict_path, 'w', encoding = 'utf-8') as fout:
        fout.write(str(sentiment_dict))


def _build_sentiment_word_dict():
    model = get_model()
    seed_dict = {}
    for emotion in emotion_list:
        seed_dict[emotion] = model.wv.most_similar(positive = [emotion], topn = 20)
    poems = Poems(emotion_poem_corpus)
    seg = Segmenter()
    poems_data = []

    for poem in poems:
        poem_data = []
        for sentence in poem:
            poem_data.extend(seg.segment(sentence))
        poems_data.append(poem_data)
    
    word2score = _tf_idf(poems_data)

    sentiment_dict = {}
    for word in word2score:
        if word not in model.wv:
            continue
        sentiment_dict[word] = []
        for emotion in emotion_list:
            score = 0.0
            for seed in seed_dict[emotion]:
                score += model.wv.similarity(seed[0], word) * seed[1] * word2score[word]
            sentiment_dict[word].append(score)
    with open(sentiment_word_dict_path, 'w', encoding = 'utf-8') as fout:
        fout.write(str(sentiment_dict))




class Sentiment():
    def __init__(self):
        if not os.path.exists(emotion_dir):
            os.mkdir(emotion_dir)
        if not os.path.exists(sentiment_dict_path):
            _build_sentiment_dict()
        if not os.path.exists(sentiment_word_dict_path):
            _build_sentiment_word_dict()
            
        with open(sentiment_dict_path, 'r', encoding = 'utf-8') as fin:
            self.sentiment_dict = eval(fin.read())
        with open(sentiment_word_dict_path, 'r', encoding = 'utf-8') as fin:
            self.sentiment_word_dict = eval(fin.read())
        
        self.seg = Segmenter()
    
    def predict_by_word(self, sentence):
        word_list = self.seg.segment(sentence)
        emotion_score = [0 for i in range(len(emotion_list))]
        for word in word_list:
            if word in self.sentiment_word_dict:
                emotion_score = [emotion_score[i] + self.sentiment_word_dict[word][i] for i in range(len(emotion_list))]
        max_prob = sorted(range(len(emotion_list)), key = lambda k : emotion_score[k])[-1]

        
        predict = emotion_list[max_prob] if max(emotion_score) > 0 else '无'


        return predict, emotion_score



    def predict(self, sentence):
        emotion_score = [0 for i in range(len(emotion_list))]
        for char in sentence:
            if char in self.sentiment_dict:
                emotion_score = [emotion_score[i] + (self.sentiment_dict[char][i] if self.sentiment_dict[char][i] > 0 else 0)
                 for i in range(len(emotion_list))]

        max_prob = sorted(range(len(emotion_list)), key = lambda k : emotion_score[k])[-1]

        
        predict = emotion_list[max_prob] if max(emotion_score) > 0 else '无'


        return predict, max_prob, emotion_score

    def analysis(self, sentence):
        word_list = self.seg.segment(sentence)
        word_list = filter(lambda w : len(w) > 1, word_list)
        word_list = [(word, self.predict(word)[0]) for word in word_list]
        return word_list







if __name__ == '__main__':
    pass