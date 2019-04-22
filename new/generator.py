#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from utils import _BATCH_SIZE, _NUM_UNITS, CHAR_VEC_DIM, NUM_OF_SENTENCES

class Generator():

    def _build_model(self):
        context_input = tf.keras.Input(shape = (None, CHAR_VEC_DIM))
        keyword_input = tf.keras.Input(shape = (None, CHAR_VEC_DIM))
        decoder_input = tf.keras.Input(shape = (None, CHAR_VEC_DIM))
        context_encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                _NUM_UNITS,
                return_state = True,
                return_sequences = True,
            ),
            input_shape = (None, CHAR_VEC_DIM)
        )
        keyword_encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                _NUM_UNITS,
                return_state = True,
                return_sequences = True,
            ),
            input_shape = (None, CHAR_VEC_DIM)
        )

        _, context_states = context_encoder(context_input)
        _, keyword_states = keyword_encoder(keyword_input)
        
        keyword_states = tf.concat((keyword_states[:1], keyword_states[-1:]), axis = 0)
        encoder_states = tf.concat((keyword_states, context_states), axis = 0)

        attention = tf.contrib.seq2seq.BahdanauAttention(
            num_units = _NUM_UNITS,
            memory = encoder_states
        )
        decoder_cell = tf.contrib.seq2seq.AttnetionWrapper(
            cell = tf.keras.layers.GRUCell(_NUM_UNITS),
            attention_mechanism = attention
        )

        decoder = tf.keras.layers.RNN(
            decoder_cell,
            return_sequences = True,
            return_state = True
        )
        decoder_outputs, _ = decoder(
            decoder_input, 
            initial_state = context_states
        )
        decoder_dense = tf.keras.layers.Dense()
