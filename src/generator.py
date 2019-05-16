#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from char2vec import Char2Vec
from char_dict import CharDict, end_of_sentence, start_of_sentence
from data_utils import batch_train_data
from paths import save_dir, write_log
from pron_dict import PronDict
from random import random
from singleton import Singleton
from utils import CHAR_VEC_DIM, NUM_OF_SENTENCES
from emotion import Sentiment, emotion_list
import numpy as np
import os
import sys
import tensorflow as tf
import time


_BATCH_SIZE = 128
_NUM_UNITS = 512
_EMOTION_DIM = 7

_model_path = os.path.join(save_dir, 'model')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Generator(Singleton):

    def _build_keyword_encoder(self):
        """ Encode keyword into a vector."""
        self.keyword = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "keyword")
        self.keyword_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "keyword_length")
        bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                cell_bw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                inputs = self.keyword,
                sequence_length = self.keyword_length,
                dtype = tf.float32, 
                time_major = False,
                scope = "keyword_encoder")

        self.keyword_outputs = tf.concat((bi_outputs[0][:, -1:,:], bi_outputs[1][:, :1, :]), axis = 2)
        tf.TensorShape([_BATCH_SIZE, 1, _NUM_UNITS]).\
                assert_same_rank(self.keyword_outputs.shape)
        

    def _build_context_encoder(self):
        """ Encode context into a list of vectors. """
        self.context = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "context")
        self.context_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "context_length")
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                cell_bw = tf.contrib.rnn.GRUCell(_NUM_UNITS / 2),
                inputs = self.context,
                sequence_length = self.context_length,
                dtype = tf.float32, 
                time_major = False,
                scope = "context_encoder")
        self.context_outputs = tf.concat(bi_outputs, axis = 2)
        self.context_states = tf.concat(bi_states, axis = 1)
        tf.TensorShape([_BATCH_SIZE, None, _NUM_UNITS]).\
                assert_same_rank(self.context_outputs.shape)
        tf.TensorShape([_BATCH_SIZE, _NUM_UNITS]).\
                assert_same_rank(self.context_states.shape)
    
    def _build_emotion_encoder(self):
        self.emotion = tf.placeholder(
                shape = [_BATCH_SIZE, _EMOTION_DIM],
                dtype = tf.float32,
                name = "emotion"
        )
        emotion_w = tf.Variable(
                tf.random_normal(shape = [_EMOTION_DIM, _NUM_UNITS],
                                mean = 0.0, stddev = 0.08), 
                trainable = True
        )
        emotion_b = tf.Variable(
                tf.random_normal(shape = [_NUM_UNITS],
                                mean = 0.0, stddev = 0.08),
                trainable = True
        )
        emotion_outputs = tf.nn.bias_add(
                tf.matmul(self.emotion, emotion_w),
                bias = emotion_b
        )
        emotion_outputs = tf.nn.tanh(emotion_outputs)
        self.emotion_outputs = tf.reshape(emotion_outputs, [_BATCH_SIZE, 1, _NUM_UNITS])
        tf.TensorShape([_BATCH_SIZE, 1, _NUM_UNITS]).\
                assert_same_rank(self.emotion_outputs.shape)

    def _build_decoder(self):
        """ Decode keyword and context into a sequence of vectors. """

        topic_outputs = tf.concat((self.emotion_outputs, self.keyword_outputs), axis = 1)
        encoder_outputs = tf.concat((topic_outputs, self.context_outputs), axis = 1)
        attention = tf.contrib.seq2seq.BahdanauAttention(
                num_units = _NUM_UNITS, 
                memory = encoder_outputs,
                memory_sequence_length = (self.context_length + 2))
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell = tf.contrib.rnn.GRUCell(_NUM_UNITS),
                attention_mechanism = attention)
        self.decoder_init_state = decoder_cell.zero_state(
                batch_size = _BATCH_SIZE, dtype = tf.float32).\
                        clone(cell_state = self.context_states)
        self.decoder_inputs = tf.placeholder(
                shape = [_BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "decoder_inputs")
        self.decoder_input_length = tf.placeholder(
                shape = [_BATCH_SIZE],
                dtype = tf.int32,
                name = "decoder_input_length")
        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                cell = decoder_cell,
                inputs = self.decoder_inputs,
                sequence_length = self.decoder_input_length,
                initial_state = self.decoder_init_state,
                dtype = tf.float32, 
                time_major = False,
                scope = "training_decoder")
        tf.TensorShape([_BATCH_SIZE, None, _NUM_UNITS]).\
                assert_same_rank(self.decoder_outputs.shape)

    def _build_projector(self):
        """ Project decoder_outputs into character space. """
        softmax_w = tf.Variable(
                tf.random_normal(shape = [_NUM_UNITS, len(self.char_dict)],
                    mean = 0.0, stddev = 0.08), 
                trainable = True)
        softmax_b = tf.Variable(
                tf.random_normal(shape = [len(self.char_dict)],
                    mean = 0.0, stddev = 0.08),
                trainable = True)
        reshaped_outputs = self._reshape_decoder_outputs()
        self.logits = tf.nn.bias_add(
                tf.matmul(reshaped_outputs, softmax_w),
                bias = softmax_b)
        self.probs = tf.nn.softmax(self.logits)

    def _reshape_decoder_outputs(self):
        """ Reshape decoder_outputs into shape [?, _NUM_UNITS]. """
        def concat_output_slices(idx, val):
            output_slice = tf.slice(
                    input_ = self.decoder_outputs,
                    begin = [idx, 0, 0],
                    size = [1, self.decoder_input_length[idx],  _NUM_UNITS])
            return tf.add(idx, 1),\
                    tf.concat([val, tf.squeeze(output_slice, axis = 0)], 
                            axis = 0)
        tf_i = tf.constant(0)
        tf_v = tf.zeros(shape = [0, _NUM_UNITS], dtype = tf.float32)
        _, reshaped_outputs = tf.while_loop(
                cond = lambda i, v: i < _BATCH_SIZE,
                body = concat_output_slices,
                loop_vars = [tf_i, tf_v],
                shape_invariants = [tf.TensorShape([]),
                    tf.TensorShape([None, _NUM_UNITS])])
        tf.TensorShape([None, _NUM_UNITS]).\
                assert_same_rank(reshaped_outputs.shape)
        return reshaped_outputs

    def _build_optimizer(self):
        """ Define cross-entropy loss and minimize it. """
        self.targets = tf.placeholder(
                shape = [None],
                dtype = tf.int32, 
                name = "targets")
        labels = tf.one_hot(self.targets, depth = len(self.char_dict))
        cross_entropy = tf.losses.softmax_cross_entropy(
                onehot_labels = labels,
                logits = self.logits)
        self.loss = tf.reduce_mean(cross_entropy)

        self.learning_rate = tf.clip_by_value(
                tf.multiply(1.6e-5, tf.pow(2.1, self.loss)),
                clip_value_min = 0.0002,
                clip_value_max = 0.02)

        self.opt_step = tf.train.AdamOptimizer(
                learning_rate = self.learning_rate).\
                        minimize(loss = self.loss)

    def _build_graph(self):
        self._build_keyword_encoder()
        self._build_context_encoder()
        self._build_emotion_encoder()
        self._build_decoder()
        self._build_projector()
        self._build_optimizer()

    def __init__(self):
        self.tf_config = tf.ConfigProto()
        # self.tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.char_dict = CharDict()
        self.char2vec = Char2Vec()
        self.sentiment = Sentiment()
        self._build_graph()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.saver = tf.train.Saver(tf.global_variables())
        self.trained = False
        
    def _initialize_session(self, session):
        
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        if not checkpoint or not checkpoint.model_checkpoint_path:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            session.run(init_op)
        else:
            self.saver.restore(session, checkpoint.model_checkpoint_path)
            self.trained = True
        pass

    def generate(self, keywords, emotion = '悲'):
        assert NUM_OF_SENTENCES == len(keywords)
        pron_dict = PronDict()
        context = start_of_sentence()
        with tf.Session(config = self.tf_config) as session:
            self._initialize_session(session)
            if not self.trained:
                print("Please train the model first! (./train.py -g)")
                sys.exit(1)
            for keyword in keywords:
                keyword_data, keyword_length = self._fill_np_matrix(
                        [keyword] * _BATCH_SIZE)
                context_data, context_length = self._fill_np_matrix(
                        [context] * _BATCH_SIZE)
                emotion_data = np.zeros(shape = [_BATCH_SIZE, _EMOTION_DIM], dtype = np.float32)
                for i in range(len(emotion_data)):
                        if emotion not in emotion_list:
                                continue
                        j = emotion_list.index(emotion)
                        emotion_data[i, j] = 1
                char = start_of_sentence()
                for _ in range(7):
                    decoder_input, decoder_input_length = \
                            self._fill_np_matrix([char])
                    encoder_feed_dict = {
                            self.keyword : keyword_data,
                            self.keyword_length : keyword_length,
                            self.context : context_data,
                            self.context_length : context_length,
                            self.emotion : emotion_data,
                            self.decoder_inputs : decoder_input,
                            self.decoder_input_length : decoder_input_length
                            }
                    if char == start_of_sentence():
                        pass
                    else:
                        encoder_feed_dict[self.decoder_init_state] = state
                    probs, state = session.run(
                            [self.probs, self.decoder_final_state], 
                            feed_dict = encoder_feed_dict)
                    prob_list = self._gen_prob_list(probs, context, pron_dict)

                #     Algorithm 1:
                #     prob_sums = np.cumsum(prob_list)
                #     rand_val = prob_sums[-1] * random()
                #     for i, prob_sum in enumerate(prob_sums):
                #         if rand_val < prob_sum:
                #             char = self.char_dict.int2char(i)
                #             break

                #     Algorithm 2:
                #     max_prob = prob_list[0]
                #     candidates = []
                #     char = self.char_dict.int2char(0)
                #     for i in range(1, len(prob_list)):
                #         if prob_list[i] >= max_prob:
                #             max_prob = prob_list[i]
                #             char = self.char_dict.int2char(i)
                #     max_prob = max_prob * (random() * 0.5 + 0.5)
                #     for i in range(len(prob_list)):
                #         if prob_list[i] >= max_prob:
                #             candidates.append(i)
                #     char_index = candidates[int(random() * len(candidates))]
                #     char = self.char_dict.int2char(char_index)

                #     Algorithm 3:
                #     prob_rank = sorted(enumerate(prob_list), key = lambda x : x[1])[-3 : -1]
                #     prob_rank = [i for i, v in prob_rank]
                #     char = self.char_dict.int2char(prob_rank[int(random() * 2)])

                #     Algorithm 4:
                #     max_prob = prob_list[0]
                #     char = self.char_dict.int2char(0)
                #     for i in range(1, len(prob_list)):
                #         if prob_list[i] >= max_prob:
                #             max_prob = prob_list[i]
                #             char = self.char_dict.int2char(i)
                #     max_prob = max_prob * (random() * 0.4 + 0.6)
                #     for i in range(len(prob_list)):
                #         if prob_list[i] >= max_prob:
                #             char = self.char_dict.int2char(i)
                #             break

                # Algorithm 5
                    max_prob = prob_list[0]
                    char = self.char_dict.int2char(0)
                    for i in range(1, len(prob_list)):
                        if prob_list[i] >= max_prob:
                            max_prob = prob_list[i]
                            char = self.char_dict.int2char(i)

                    context += char
                context += end_of_sentence()
        return context[1:].split(end_of_sentence())

    def _gen_prob_list(self, probs, context, pron_dict):
        prob_list = probs.tolist()[0]
        prob_list[0] = 0
        prob_list[-1] = 0
        idx = len(context)
        # error_words = ['一','不','何','无','上','未','处','己','中','已','多','开','人','生','树']
        error_words = ['一','不','何','无','上','未','处','己','中','已']
        used_chars = set(ch for ch in context)
        for i in range(1, len(prob_list) - 1):
            ch = self.char_dict.int2char(i)
            # Penalize used characters.
            if ch in used_chars:
                prob_list[i] *= 0.0
            # Penalize rhyming violations.
            if (idx == 15 or idx == 31) and \
                    not pron_dict.co_rhyme(ch, context[7]):
                prob_list[i] *= 0.0
            # Penalize tonal violations.
            if idx > 2 and 2 == idx % 8 and \
                    not pron_dict.counter_tone(context[2], ch):
                prob_list[i] *= 0.0
            if (4 == idx % 8 or 6 == idx % 8) and \
                    not pron_dict.counter_tone(context[idx - 2], ch):
                prob_list[i] *= 0.0
            if (ch in error_words) and idx < 8:
                prob_list[i] *= 0.0
        return prob_list

    def train(self, n_epochs = 6):
        
        with tf.Session(config = self.tf_config) as session:
            self._initialize_session(session)
            try:
                print("Training RNN-based generator ...")
                write_log("[%s] Generator Training begin !" % time.ctime())
                for epoch in range(n_epochs):
                    batch_no = 0
                    loss = 0.0
                    learning_rate = 0.0
                    beg_time = time.time()
                    now_time = time.time()
                    for keywords, contexts, sentences, total \
                            in batch_train_data(_BATCH_SIZE):

                       
                        tem_loss,tem_lr = self._train_a_batch(session, epoch,
                                keywords, contexts, sentences)
                        loss += tem_loss
                        learning_rate += tem_lr
                        batch_no += 1
                        now_time = time.time()

                        time.sleep(0.1)
                        print(
                                '[Model Training] total %d, process is %d%%, time used %d(s)' % (total, batch_no *  _BATCH_SIZE / total * 100, now_time - beg_time), end = '\r'
                        )
                    log_string = '[%s] [Generator] epoch = %d, loss = %f, learning_rate = %f, time used %f(s)' % (time.ctime(), epoch + 1, loss / batch_no, learning_rate / batch_no, now_time - beg_time)
                    self.saver.save(session, _model_path)
                    print(log_string)
                    write_log(log_string)
                    time.sleep(60)
                    
                print("Training is done.")
            except KeyboardInterrupt:
                print("Training is interrupted.")

    def _train_a_batch(self, session, epoch, keywords, contexts, sentences):
        keyword_data, keyword_length = self._fill_np_matrix(keywords)
        context_data, context_length = self._fill_np_matrix(contexts)
        decoder_inputs, decoder_input_length  = self._fill_np_matrix(
                [start_of_sentence() + sentence[:-1] \
                        for sentence in sentences])
        targets = self._fill_targets(sentences)

        emotions = [self.sentiment.predict(sentence)[0] for sentence in sentences]
        matrix = np.zeros([_BATCH_SIZE, _EMOTION_DIM], dtype = np.float32)
        for i in range(len(emotions)):
                if emotions[i] not in emotion_list:
                        continue
                j = emotion_list.index(emotions[i])
                matrix[i, j] = 1
        feed_dict = {
                self.keyword : keyword_data,
                self.keyword_length : keyword_length,
                self.context : context_data,
                self.context_length : context_length,

                self.emotion : matrix,

                self.decoder_inputs : decoder_inputs,
                self.decoder_input_length : decoder_input_length,
                self.targets : targets
                }
        
        loss, learning_rate, _ = session.run(
                [self.loss, self.learning_rate, self.opt_step],
                feed_dict = feed_dict)

        return loss, learning_rate

    def _fill_np_matrix(self, texts):
        max_time = max(map(len, texts))
        matrix = np.zeros([_BATCH_SIZE, max_time, CHAR_VEC_DIM], 
                dtype = np.float32)
        for i in range(_BATCH_SIZE):
            for j in range(max_time):
                matrix[i, j, :] = self.char2vec.get_vect(end_of_sentence())
        for i, text in enumerate(texts):
            matrix[i, : len(text)] = self.char2vec.get_vects(text)
        seq_length = [len(texts[i]) if i < len(texts) else 0 \
                for i in range(_BATCH_SIZE)]
        return matrix, seq_length

    def _fill_targets(self, sentences):
        targets = []
        for sentence in sentences:
            targets.extend(map(self.char_dict.char2int, sentence))
        return targets


if __name__ == '__main__':
    generator = Generator()
    keywords = ['四时', '变', '雪', '新']
    poem = generator.generate(keywords)
    for sentence in poem:
        print(sentence)

