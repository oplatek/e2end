#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . import E2E_property_decodingBase 
import tensorflow as tf
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FastComp(E2E_property_decodingBase):
    '''Dummy class just for debugging training loop - it compiles fast.'''
    def __init__(self, config):
        self.config = c = config
        self._var2save = tf.Variable([1])
        self.step, self.config = 0, config

        self._define_inputs()
        arr = [
                  self.turn_len, self.dec_targets, self.target_lens,
                  self.is_first_turn, self.feed_previous,
                  self.dropout_keep_prob, self.dropout_db_keep_prob,
              ] + self.feat_list
        self.testTrainOp = tf.concat(0, [tf.to_float(tf.reshape(x, (-1, 1))) for x in arr])

    def train_step(self, sess, train_dict, log_output=False):
        self.step += 1
        sess.run(self.testTrainOp, train_dict)
        logger.debug('train_dict: %s', train_dict)
        logger.debug('input_feed_dict_shape: %s', [(k, v.shape) if hasattr(v, 'shape') else (k, v) for k, v in train_dict.items()])

        if log_output:
            return {'reward': -666, 'summarize': tf.Summary(value=[tf.Summary.Value(tag='dummy_loss', simple_value=-666)])}
        else:
            return {}
        return {}

    def decode_step(self, sess, input_feed_dict):
        logger.debug('input_feed_dict: %s', input_feed_dict)
        return {'decoder_outputs': [[0]], 'loss': -777}

    def eval_step(self, sess, labels_dict, log_output=False):
        return {'decoder_outputs': [[1]], 'loss': -777,
                'summarize': tf.Summary(value=[tf.Summary.Value(tag='dummy_loss', simple_value=-666)]),
                'reward': -888}
