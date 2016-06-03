#!/usr/bin/env python
# -*- coding: utf-8 -*-
from . import E2E_property_decodingBase
import tensorflow as tf
import logging
from .evaluation import tf_lengths2mask2d

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SumRows(E2E_property_decodingBase):

    def _build_db(self, col_embeddings, encoder_cell, words_hidden_feat, dialog_state_after_turn, words_embedded):
        c = self.config
        db_rows_embeddings_a = []
        for i in range(c.num_rows):
            row_embed_arr = [
                tf.nn.embedding_lookup(col_embeddings[j], self.db_rows[i, j]) for j in
                range(c.num_cols)]
            row_embed = tf.expand_dims(tf.concat(0, row_embed_arr), -1)
            i or logger.debug('row_embed_arr is list of different [%s] * %d', row_embed_arr[0].get_shape(),
                              len(row_embed_arr))
            i or logger.debug('row_embed shape %s', row_embed.get_shape())
            db_rows_embeddings_a.append(row_embed)
        db_rows_embeddings = tf.concat(0, db_rows_embeddings_a)
        summed_rows = tf.reduce_sum(db_rows_embeddings, 0)
        batched_summed_rows = tf.tile(tf.expand_dims(summed_rows, 0), [c.batch_size, 1])
        return batched_summed_rows 


class EmbeddingMatchDB(E2E_property_decodingBase):
    def _build_db(self, col_embeddings, encoder_cell, words_hidden_feat, dialog_state_after_turn, words_embedded):
        c = self.config

        logger.debug('Building word & db_vocab attention started %.2f from DB construction start')
        logger.debug('Simple computation because lot of inputs')
        assert c.col_emb_size == encoder_cell.hidden_size, 'Otherwise I cannot do scalar product'

        logger.debug('dbembeddings is a matrix embedding_size x db_vocab; db_vocab group_by collumn')
        dbembeddings = tf.concat(0, col_embeddings)
        batched_dbembeddings = tf.tile(tf.expand_dims(dbembeddings, 0), [c.batch_size, 1])
        logger.debug('words_hgeat_mat is batch x matrix max_turn_len x embedding_size; the last column may be gardbage bacause turn is shorter')
        words_hfeat_mat = tf.concat(1, words_hidden_feat)
        w_norm = tf.sqrt(tf.reduce_sum(tf.square(words_hfeat_mat), 1, keep_dims=True))
        norm_words_hfeat = words_hfeat_mat / w_norm
        logger.debug('norm_batch_dbemb has shape batch x col x embedding_size')
        e_norm = tf.sqrt(tf.reduce_sum(tf.square(batched_dbembeddings), 1, keep_dims=True))
        norm_batch_dbemb = batched_dbembeddings / e_norm  
        logger.debug('cosine_sim: batch x c.max_turn_len x num_db_ent')
        cosine_sim = tf.batch_matmul(norm_words_hfeat, tf.transpose(norm_batch_dbemb, perm=[0, 2, 1]))

        logger.debug('words_mask: batch x c.max_turn_len')
        words_mask = tf_lengths2mask2d(self.turn_len, c.max_turn_len)
        logger.debug('broadcasted words_mask')
        masked_cos_sim = tf.mul(cosine_sim, words_mask)
        logger.debug('row_sim: batch x num_db_vocab')
        db_entity_att = tf.reduce_mean(masked_cos_sim, 1)
        tf.image_summary('db_entity_att', tf.expand_dims(tf.expand_dims(db_entity_att, -1), -1), collections=['images'], max_images=c.batch_size)

        db_rows_embeddings = []
        for i in range(c.num_rows):
            row_embed_arr = [
                tf.nn.embedding_lookup(col_embeddings[j], self.db_rows[i, j]) for j in
                range(c.num_cols)]
            row_embed = tf.concat(0, row_embed_arr)
            i or logger.debug('row_embed_arr is list of different [%s] * %d', row_embed_arr[0].get_shape(),
                              len(row_embed_arr))
            i or logger.debug('row_embed shape %s', row_embed.get_shape())
            db_rows_embeddings.append(row_embed)

        # for i in range(c.num_rows):
        #     tf.gather(cosine_sim, self.db_rows[i, ;])
        # tf.gather( self.db_rows[i, j]

        # db_embed_concat = tf.concat(1, [tf.squeeze(row_selected, [2]), db_embed])
        # return db_embed_concat 
