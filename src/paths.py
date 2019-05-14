#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import os


root_dir = os.path.dirname(__file__)
data_dir = os.path.join(root_dir, 'data')
raw_dir = os.path.join(root_dir, 'raw')
save_dir = os.path.join(root_dir, 'save')
log_dir = os.path.join(root_dir, 'log')

sxhy_path = os.path.join(data_dir, 'sxhy_dict.txt')
char_dict_path = os.path.join(data_dir, 'char_dict.txt')
poems_path = os.path.join(data_dir, 'poem.txt')
char2vec_path = os.path.join(data_dir, 'char2vec.npy')
wordrank_path = os.path.join(data_dir, 'wordrank.txt')
plan_data_path = os.path.join(data_dir, 'plan_data.txt')
gen_data_path = os.path.join(data_dir, 'gen_data.txt')

plan_model_path = os.path.join(save_dir, 'plan_model.bin')
expander_model_path = os.path.join(save_dir, 'expander.h5')
# Emotion:

emotion_dir = os.path.join(root_dir, 'emotion')

emotion_poem_corpus = os.path.join(emotion_dir, 'emotion_poems.txt')
sentiment_dict_path = os.path.join(emotion_dir, 'sentiment_dict.txt')
sentiment_word_dict_path = os.path.join(emotion_dir, 'sentiment_word_dict.txt')
all_poems_w2v_model_path = os.path.join(emotion_dir, 'all_poems_w2v.model')

log_path = os.path.join(log_dir, 'log.txt')

# TODO: configure dependencies in another file.
_dependency_dict = {
        poems_path : [char_dict_path],
        char2vec_path : [char_dict_path, poems_path],
        wordrank_path : [sxhy_path, poems_path],
        gen_data_path : [char_dict_path, poems_path, sxhy_path, char2vec_path],
        plan_data_path : [char_dict_path, poems_path, sxhy_path, char2vec_path],

        }

def check_uptodate(path):
    """ Return true iff the file exists and up-to-date with dependencies."""
    if not os.path.exists(path):
        # File not found.
        return False
    timestamp = os.path.getmtime(path)
    if path in _dependency_dict:
        for dependency in _dependency_dict[path]:
            if not os.path.exists(dependency) or \
                    os.path.getmtime(dependency) > timestamp:
                # File stale.
                return False
    return True

def write_log(string, path = log_path):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    with open(log_path, 'a', encoding = 'utf-8') as fout:
        fout.write(string + '\n')