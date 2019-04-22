#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from utils import _BATCH_SIZE, _NUM_UNITS, CHAR_VEC_DIM, NUM_OF_SENTENCES

class Generator():

    def _build_model(self):
        context_encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                _NUM_UNITS,
                return_state = True,
                return_sequences = True,
            ),
            input_shape = (_BATCH_SIZE, None, CHAR_VEC_DIM)
        )
        keyword_encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                _NUM_UNITS,
                return_state = True,
                return_sequences = True,
            ),
            input_shape = (_BATCH_SIZE, None, CHAR_VEC_DIM)
        )
        decoder = tf.keras.layers.GRU(
            _NUM_UNITS,
            return_state = True,
        )