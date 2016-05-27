#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . import E2E_property_decoding
import tensorflow as tf
import numpy as np
import logging
from ..utils import elapsed_timer
from .evaluation import tf_trg_word2vocab_id, tf_lengths2mask2d, get_bleus

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RowPredictions(E2E_property_decoding):

    def __init__(self, config):
        self.config = c = config  # shortcut as it is used heavily
        logger.info('Compiling %s', self.__class__.__name__)
        self.step = 0

        self._define_inputs(c)
        with tf.variable_scope('encoder'), elapsed_timer() as inpt_timer:
            encoder_cell, words_hidden_feat, dialog_state_after_turn, words_embedded, words_embeddings = self._build_encoder(c)
            logger.debug('Initialization of encoder took  %.2f s.', inpt_timer())

        with tf.variable_scope('db_encoder'), elapsed_timer() as db_timer:
            if c.use_db_encoder:
                db_embed, row_selected, col_embeddings = self._build_db(c, encoder_cell, words_hidden_feat, dialog_state_after_turn, words_embedded)
                encoded_state = tf.concat(1, [dialog_state_after_turn, tf.squeeze(row_selected, [2]), db_embed])
                att_hidd_feat_list = words_hidden_feat + [db_embed]
                logger.info('\nInitialized db encoder in %.2f s\n', db_timer())
            else:
                col_embeddings = None
                encoded_state = dialog_state_after_turn
                att_hidd_feat_list = words_hidden_feat
                logger.info('\nUsing plain encoder decoder\n')

        with tf.variable_scope('decoder'), elapsed_timer() as dec_timer:
            targets, target_mask, dec_logitss = self._build_decoder(c, encoded_state, att_hidd_feat_list, col_embeddings, words_embeddings)
            self.dec_outputs = [tf.arg_max(dec_logits, 1) for dec_logits in dec_logitss]
            logger.debug('Building of the decoder took %.2f s.', dec_timer())

