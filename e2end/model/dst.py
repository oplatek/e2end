#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . import E2E_property_decodingBase
import tensorflow as tf
import numpy as np
import logging
from ..utils import elapsed_timer
from .evaluation import tf_lengths2mask2d

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DstIndep(E2E_property_decodingBase):
    def __init__(self, config):
        self.config = c = config  # shortcut as it is used heavily
        logger.info('Compiling %s', self.__class__.__name__)
        self.step = 0
        self.last_reward_val = 0

        self._define_inputs()
        with tf.variable_scope('encoder'), elapsed_timer() as inpt_timer:
            encoder_cell, words_hidden_feat, dialog_state_after_turn, words_embedded, words_embeddings = self._build_encoder()
            logger.debug('Initialization of encoder took  %.2f s.', inpt_timer())

        with tf.variable_scope('decoder'), elapsed_timer() as dec_timer:

            logger.debug('The decoder uses special token GO_ID as first input. Adding to vocabulary.')
            self.GO_ID = c.num_words
            self.goid = tf.constant(self.GO_ID)
            goid_batch_vec = tf.constant([self.GO_ID] * c.batch_size, shape=(c.batch_size, 1), dtype=tf.int64)
            logger.debug('Adding GO_ID at the beggining of each decoder input')
            decoder_inputs2D = [goid_batch_vec] + tf.split(1, c.max_target_len, self.dec_targets)
            targets = [tf.squeeze(di, [1]) for di in decoder_inputs2D]
            targets= targets[1:]  # we have inserted GO_ID symbol
            logger.debug('targets[0:1].get_shape(): %s, %s', targets[0].get_shape(), targets[1].get_shape())
            self.target_mask = [tf.squeeze(m, [1]) for m in tf.split(1, c.max_target_len, tf_lengths2mask2d(self.target_lens, c.max_target_len))]

            state_size = np.prod(dialog_state_after_turn.get_shape().as_list()[1:])

            num_decoded = sum(c.col_vocab_sizes + [c.num_words])
            m1_out = tf.get_variable('dst_m1_out', initializer=tf.random_normal([state_size, num_decoded]))
            b1_out = tf.get_variable('dst_b1_out', initializer=tf.random_normal([num_decoded]))
            first = tf.nn.softmax(tf.nn.xw_plus_b(dialog_state_after_turn, m1_out, b1_out))

            m2_out = tf.get_variable('dst_m2_out', initializer=tf.random_normal([state_size, num_decoded]))
            b2_out = tf.get_variable('dst_b2_out', initializer=tf.random_normal([num_decoded]))
            second = tf.nn.softmax(tf.nn.xw_plus_b(dialog_state_after_turn, m2_out, b2_out))

            m3_out = tf.get_variable('dst_m3_out', initializer=tf.random_normal([state_size, num_decoded]))
            b3_out = tf.get_variable('dst_b3_out', initializer=tf.random_normal([num_decoded]))
            third = tf.nn.softmax(tf.nn.xw_plus_b(dialog_state_after_turn, m3_out, b3_out))

            dec_logitss = [first, second, third] + ([c.EOS_ID * tf.ones((c.batch_size, num_decoded))] * (len(targets) - 3))

            targets, self.target_mask, dec_logitss = self._build_decoder(encoder_cell, dialog_state_after_turn, words_embedded, words_embeddings, [])
            self.dec_outputs = [tf.arg_max(dec_logits, 1) for dec_logits in dec_logitss]
            logger.debug('Building of the decoder took %.2f s.', dec_timer())

        with tf.variable_scope('loss_and_eval'), elapsed_timer() as loss_timer:
            self.loss, self.eval_functions = self._build_reward_func(dec_logitss, targets)
            logger.debug('Building the loss/reward functions ops took %.2f s', loss_timer())

        with tf.variable_scope('updates'), elapsed_timer() as updates_timer:
            self._optimizer = opt = tf.train.AdamOptimizer(c.learning_rate)
            tf.scalar_summary(self.loss.op.name + 'loss', self.loss)
            params = tf.trainable_variables()
            gradients = opt.compute_gradients(self.loss, params) 
            modified_grads = gradients
            self.xent_updates = opt.apply_gradients(modified_grads)
            logger.debug('Building the gradient udpate ops took %.2f s', updates_timer())

        self.summarize = tf.merge_all_summaries()

        times = [inpt_timer(), dec_timer(), loss_timer(), updates_timer()]
        logger.debug('Blocks times: %s,\n total: %.2f', times, sum(times))


class DstJoint(E2E_property_decodingBase):
    def __init__(self, config):
        self.config = c = config  # shortcut as it is used heavily
        logger.info('Compiling %s', self.__class__.__name__)
        self.step = 0
        self.last_reward_val = 0

        self._define_inputs()
        with tf.variable_scope('encoder'), elapsed_timer() as inpt_timer:
            encoder_cell, words_hidden_feat, dialog_state_after_turn, words_embedded, words_embeddings = self._build_encoder()
            logger.debug('Initialization of encoder took  %.2f s.', inpt_timer())

        with tf.variable_scope('decoder'), elapsed_timer() as dec_timer:

            logger.debug('The decoder uses special token GO_ID as first input. Adding to vocabulary.')
            self.GO_ID = c.num_words
            self.goid = tf.constant(self.GO_ID)
            goid_batch_vec = tf.constant([self.GO_ID] * c.batch_size, shape=(c.batch_size, 1), dtype=tf.int64)
            logger.debug('Adding GO_ID at the beggining of each decoder input')
            decoder_inputs2D = [goid_batch_vec] + tf.split(1, c.max_target_len, self.dec_targets)
            targets = [tf.squeeze(di, [1]) for di in decoder_inputs2D]
            targets= targets[1:]  # we have inserted GO_ID symbol
            logger.debug('targets[0:1].get_shape(): %s, %s', targets[0].get_shape(), targets[1].get_shape())
            self.target_mask = [tf.squeeze(m, [1]) for m in tf.split(1, c.max_target_len, tf_lengths2mask2d(self.target_lens, c.max_target_len))]

            state_size = np.prod(dialog_state_after_turn.get_shape().as_list()[1:])

            num_decoded = sum(c.col_vocab_sizes + [c.num_words])
            m1_out = tf.get_variable('dst_m1_out', initializer=tf.random_normal([state_size, num_decoded]))
            b1_out = tf.get_variable('dst_b1_out', initializer=tf.random_normal([num_decoded]))
            first = tf.nn.softmax(tf.nn.xw_plus_b(dialog_state_after_turn, m1_out, b1_out))

            m2_out = tf.get_variable('dst_m2_out', initializer=tf.random_normal([state_size, num_decoded]))
            b2_out = tf.get_variable('dst_b2_out', initializer=tf.random_normal([num_decoded]))
            second = tf.nn.softmax(tf.nn.xw_plus_b(dialog_state_after_turn, m2_out, b2_out))

            m3_out = tf.get_variable('dst_m3_out', initializer=tf.random_normal([state_size, num_decoded]))
            b3_out = tf.get_variable('dst_b3_out', initializer=tf.random_normal([num_decoded]))
            third = tf.nn.softmax(tf.nn.xw_plus_b(dialog_state_after_turn, m3_out, b3_out))

            dec_logitss = [first, second, third] + ([c.EOS_ID * tf.ones((c.batch_size, num_decoded))] * (len(targets) - 3))

            targets, self.target_mask, dec_logitss = self._build_decoder(encoder_cell, dialog_state_after_turn, words_embedded, words_embeddings, [])
            self.dec_outputs = [tf.arg_max(dec_logits, 1) for dec_logits in dec_logitss]
            logger.debug('Building of the decoder took %.2f s.', dec_timer())

        with tf.variable_scope('loss_and_eval'), elapsed_timer() as loss_timer:
            self.loss, self.eval_functions = self._build_reward_func(dec_logitss, targets)
            logger.debug('Building the loss/reward functions ops took %.2f s', loss_timer())

        with tf.variable_scope('updates'), elapsed_timer() as updates_timer:
            self._optimizer = opt = tf.train.AdamOptimizer(c.learning_rate)
            tf.scalar_summary(self.loss.op.name + 'loss', self.loss)
            params = tf.trainable_variables()
            gradients = opt.compute_gradients(self.loss, params) 
            modified_grads = gradients
            self.xent_updates = opt.apply_gradients(modified_grads)
            logger.debug('Building the gradient udpate ops took %.2f s', updates_timer())

        self.summarize = tf.merge_all_summaries()

        times = [inpt_timer(), dec_timer(), loss_timer(), updates_timer()]
        logger.debug('Blocks times: %s,\n total: %.2f', times, sum(times))
