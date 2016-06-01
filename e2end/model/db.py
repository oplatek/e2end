#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . import E2E_property_decodingBase
import tensorflow as tf
import logging
from ..utils import elapsed_timer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Simple(E2E_property_decodingBase):

    def _build_db(self, col_embeddings, encoder_cell, words_hidden_feat, dialog_state_after_turn, words_embedded):
        c = self.config
        with elapsed_timer() as db_embed_timer:
            db_rows_embeddings = []
            for i in range(c.num_rows):
                row_embed_arr = [
                    tf.nn.dropout(tf.nn.embedding_lookup(col_embeddings[j], self.db_rows[i, j]), self.dropout_db_keep_prob) for j in
                    range(c.num_cols)]
                row_embed = tf.concat(0, row_embed_arr)
                i or logger.debug('row_embed_arr is list of different [%s] * %d', row_embed_arr[0].get_shape(),
                                  len(row_embed_arr))
                i or logger.debug('row_embed shape %s', row_embed.get_shape())
                db_rows_embeddings.append(row_embed)
            logger.debug('Create embeddings of  all db rows took %.2f s', db_embed_timer())

        batched_db_rows_embeddings = [tf.tile(tf.expand_dims(re, 0), [c.batch_size, 1]) for re in
                                      db_rows_embeddings]

        # FIXME use words_hidden_feat and words_embedded
        # logger.debug('Building word & db_vocab attention started %.2f from DB construction start')
        # logger.debug('Simple computation because lot of inputs')
        # assert c.col_emb_size == encoder.hidden_size, 'Otherwise I cannot do scalar product'
        # words_attributes_distance = []
        # for w_hid_feat in words_hidden_feat:
        #     vocab_offset = 0
        #     for vocab_emb, vocab_size in zip(col_embeddings, col_vocab_sizes):
        #         for idx in range(vocab_size):
        #             w_emb = tf.nn.embedding_lookup(vocab_emb, idx)
        #             wT_times_db_emb_value = tf.matmul(tf.transpose(w_hid_feat), w_emb)
        #             words_attributes_distance.append(wT_times_db_emb_value)
        # words_attributes_att = tf.softmax(tf.concat(0, words_attributes_distance))
        # words_vocab_entries_att = todo_reshape_into_word_TIMES_slot_vocab_TIMES_slot_value

        def select_row(batched_row, encoded_history, reuse=False, scope='select_row'):
            with tf.variable_scope(scope, reuse=reuse):
                inpt = tf.concat(1, [batched_row, encoded_history])
                inpt_size = inpt.get_shape().as_list()[1]
                W1 = tf.get_variable('W1', initializer=tf.random_normal([inpt_size, c.mlp_db_l1_size]))
                b1 = tf.get_variable('b1', initializer=tf.random_normal([c.mlp_db_l1_size]))
                layer1 = tf.nn.relu(tf.nn.xw_plus_b(inpt, W1, b1))

                last_layer = layer1
                last_layer_size = c.mlp_db_l1_size
                Out = tf.get_variable('Out', initializer=tf.random_normal([last_layer_size, 1]))
                b_out = tf.get_variable('b_out', initializer=tf.random_normal([1]))
                should_be_selected_att = tf.nn.sigmoid(tf.nn.xw_plus_b(last_layer, Out, b_out))
                return should_be_selected_att

        row_selected_arr = []
        for i, r in enumerate(batched_db_rows_embeddings):
            reuse = False if i == 0 else True
            row_sel = select_row(r, dialog_state_after_turn, reuse=reuse)
            i or logger.debug('First row selected shape: %s', row_sel.get_shape())
            row_selected_arr.append(row_sel)

        row_selected = tf.transpose(tf.pack(row_selected_arr), perm=(1, 0, 2))
        logger.debug('row_selected shape: %s', row_selected.get_shape())

        weighted_rows = [tf.mul(w, r) for w, r in zip(batched_db_rows_embeddings, row_selected_arr)]
        logger.debug('weigthed_rows[0].get_shape() %s', weighted_rows[0].get_shape())
        db_embed_inputs = tf.concat(1, weighted_rows + [dialog_state_after_turn])
        logger.debug('db_embed_inputs.get_shape() %s', db_embed_inputs.get_shape())

        input_len = db_embed_inputs.get_shape().as_list()[1]
        Wdb = tf.get_variable('Wdb', initializer=tf.random_normal([input_len, c.mlp_db_embed_l1_size]))
        logger.debug('Wdb.get_shape() %s', Wdb.get_shape())
        Bdb = tf.get_variable('Bdb', initializer=tf.random_normal([c.mlp_db_embed_l1_size]))
        l1 = tf.nn.relu(tf.nn.xw_plus_b(db_embed_inputs, Wdb, Bdb))
        out_size = encoder_cell.output_size
        WOutDb = tf.get_variable('WOutDb', initializer=tf.random_normal([c.mlp_db_embed_l1_size, out_size]))
        BOutDb = tf.get_variable('BOutDb', initializer=tf.random_normal([out_size]))
        db_embed = tf.nn.xw_plus_b(l1, WOutDb, BOutDb)
        # FIXME try to interpret the output of the DB ege again as attention
        logger.debug('db_embed.get_shape() %s', db_embed.get_shape())
        return db_embed, row_selected
