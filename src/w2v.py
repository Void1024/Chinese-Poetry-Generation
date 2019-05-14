#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from paths import all_poems_w2v_model_path, emotion_poem_corpus
from gensim import models
from poems import Poems
from segment import Segmenter
import os

def _gen_word_model():
    segment = Segmenter()
    poems = Poems(emotion_poem_corpus)
    poems_data = []
    for poem in poems:
        poem_data = []
        for sentence in poem:
            poem_data.extend(segment.segment(sentence))
        poems_data.append(poem_data)
    model = models.Word2Vec(poems_data, size = 512)
    model.save(all_poems_w2v_model_path)

def get_model(mode = 'word', content = 'all'):
    if mode == 'word' and content == 'all':
        if not os.path.exists(all_poems_w2v_model_path):
            _gen_word_model()
        model = models.Word2Vec.load(all_poems_w2v_model_path)
        return model