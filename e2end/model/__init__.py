#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import logging, math
from tensorflow.python.ops import control_flow_ops
from ..utils import elapsed_timer, sigmoid, time2batch, trim_decoded
from .decoder import embedding_attention_decoder, word_db_embed_attention_decoder
from .evaluation import tf_trg_word2vocab_id, tf_lengths2mask2d, get_bleus
from tensorflow.python.training.moving_averages import assign_moving_average

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class E2E_property_decoding():

    def _define_inputs(self, c):
        self.db_row_initializer = tf.placeholder(tf.int64, shape=(c.num_rows, c.num_cols), name='row_initializer')
        self.db_rows = tf.Variable(self.db_row_initializer, trainable=False, collections=[], name='db_rows')

        self.vocabs_cum_start_initializer = tf.placeholder(tf.int64, shape=(c.num_cols + 1,), name='vocabs_cum_start_initializer')
        self.vocabs_cum_start_idx_low = tf.Variable(self.vocabs_cum_start_initializer, trainable=False, collections=[], name='vocabs_cum_start_idx_low')
        self.vocabs_cum_start_idx_up = tf.Variable(self.vocabs_cum_start_initializer, trainable=False, collections=[], name='vocabs_cum_start_idx_up')

        self.turn_len = tf.placeholder(tf.int64, shape=(c.batch_size,), name='turn_len')

        logger.debug('Words are the most important features')
        self.feat_list = feat_list = [tf.placeholder(tf.int64, shape=(c.batch_size, c.max_turn_len), name='words')]
        logger.debug('Indicators if a word belong to a slot - like abstraction - densify data')
        for i, n in enumerate(c.column_names):
            feat = tf.placeholder(tf.int64, shape=(c.batch_size, c.max_turn_len), name='slots_indicators{}-{}'.format(i, n))
            feat_list.append(feat)
        logger.debug('Another feature for each word we have speaker id')
        self.speakerId = tf.placeholder(tf.int64, shape=(c.batch_size, c.max_turn_len), name='speakerId')
        feat_list.append(self.speakerId)

        self.dropout_keep_prob = tf.placeholder('float', name='dropout_keep_prob')
        self.dropout_db_keep_prob = tf.placeholder('float', name='dropout_db_keep_prob')

        self.is_first_turn = tf.placeholder(tf.bool, name='is_first_turn')
        self.feed_previous = tf.placeholder(tf.bool, name='feed_previous')

        self.dec_targets = tf.placeholder(tf.int64, shape=(c.batch_size, c.max_target_len), name='dec_targets')
        self.target_lens = tf.placeholder(tf.int64, shape=(c.batch_size,), name='target_lens')

    def _build_encoder(self, c):
        logger.debug('For each word_i from i in 1..max_turn_len there is a list of features: word, belongs2slot1, belongs2slot2, ..., belongs2slotK')
        logger.debug('Feature list uses placelhoder.name to create feed dictionary')

        esingle_cell = tf.nn.rnn_cell.GRUCell(c.encoder_size)
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [esingle_cell] * c.encoder_layers) if c.encoder_layers > 1 else esingle_cell

        logger.debug(
            'embedded_inputs is a list of size c.max_turn_len with tensors of shape (batch_size, all_feature_size)')

        feat_embeddings = []
        for i in range(len(self.feat_list)):
            if i == 0:  # words
                logger.debug('Increasing the size to fit in the GO_ID id. See above')
                voclen = c.num_words + 1
                emb_size = c.word_embed_size
            else:  # binary features
                voclen = 2
                emb_size = c.feat_embed_size
            feat_embeddings.append(tf.get_variable('feat_embedding{}'.format(i),
                                                   initializer=tf.random_uniform([voclen, emb_size], -math.sqrt(3),
                                                                                 math.sqrt(3))))

        embedded_inputs, embedded_words = [], []
        for j in range(c.max_turn_len):
            features_j_word = []
            for i in range(len(self.feat_list)):
                embedded = tf.nn.embedding_lookup(feat_embeddings[i], self.feat_list[i][:, j])  # FIXME it seems lookup support loading the embeddings at once for all feat_list
                dropped_embedded = tf.nn.dropout(embedded, self.dropout_keep_prob)
                if i == 0:  # words
                    embedded_words.append(dropped_embedded)
                features_j_word.append(dropped_embedded)
            w_features = tf.concat(1, features_j_word)
            j or logger.debug('Word features has shape (batch, concat_embs) == %s', w_features.get_shape())
            embedded_inputs.append(w_features)

        logger.debug(
            'We get input features for each turn, to represent dialog, we need to store the state between the turns')
        dialog_state_before_acc = tf.get_variable('dialog_state_before_turn',
                                                  initializer=tf.zeros([c.batch_size, c.encoder_size],
                                                                       dtype=tf.float32), trainable=False)
        dialog_state_before_turn = control_flow_ops.cond(self.is_first_turn,
                                                         lambda: encoder_cell.zero_state(c.batch_size, tf.float32),
                                                         lambda: dialog_state_before_acc)
        words_hidden_feat, dialog_state_after_turn = tf.nn.rnn(encoder_cell, embedded_inputs,
                                                               initial_state=dialog_state_before_turn,
                                                               sequence_length=self.turn_len)
        dialog_state_before_acc = tf.assign(dialog_state_before_acc, dialog_state_after_turn)
        words_embeddings = feat_embeddings[0]
        return encoder_cell, words_hidden_feat, dialog_state_after_turn, embedded_words, words_embeddings

    def _build_db(self, c, encoder_cell, words_hidden_feat, dialog_state_after_turn, words_embedded):
        col_embeddings = [tf.get_variable('col_values_embedding{}'.format(i),
                                          initializer=tf.random_uniform([col_vocab_size, c.col_emb_size],
                                                                        -math.sqrt(3), math.sqrt(3))) for
                          i, col_vocab_size in enumerate(c.col_vocab_sizes)]

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
        return db_embed, row_selected, col_embeddings

    def _build_decoder(self, c, encoded_state, att_hidd_feat_list, col_embeddings, word_embeddings):
        logger.debug('The decoder uses special token GO_ID as first input. Adding to vocabulary.')
        self.GO_ID = c.num_words
        self.goid = tf.constant(self.GO_ID)
        goid_batch_vec = tf.constant([self.GO_ID] * c.batch_size, shape=(c.batch_size, 1), dtype=tf.int64)
        logger.debug('Adding GO_ID at the beggining of each decoder input')
        decoder_inputs2D = [goid_batch_vec] + tf.split(1, c.max_target_len, self.dec_targets)
        targets = [tf.squeeze(di, [1]) for di in decoder_inputs2D]
        logger.debug('targets[0:1].get_shape(): %s, %s', targets[0].get_shape(), targets[1].get_shape())

        decoder_size = encoded_state.get_shape().as_list()[1]
        dsingle_cell = tf.nn.rnn_cell.GRUCell(decoder_size)
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [dsingle_cell] * c.decoder_layers) if c.decoder_layers > 1 else dsingle_cell
        target_mask = [tf.squeeze(m, [1]) for m in tf.split(1, c.max_target_len, tf_lengths2mask2d(self.target_lens, c.max_target_len))]

        # Take from tf/python/ops/seq2seq.py:706
        logger.debug('att_hidd_feat_list[0].get_shape() %s', att_hidd_feat_list[0].get_shape())
        top_states = [tf.reshape(e, [-1, 1, c.encoder_size])
                      for e in att_hidd_feat_list]  # FIXME should I change it different size that encoder_(feat) size?
        attention_states = tf.concat(1, top_states)
        logger.debug('attention_states.get_shape() %s', attention_states.get_shape())

        total_input_vocab_size = sum(c.col_vocab_sizes + [c.num_words])
        decoder_cell = tf.nn.rnn_cell.OutputProjectionWrapper(decoder_cell, total_input_vocab_size)

        encoded_state_size = encoded_state.get_shape().as_list()[1]
        assert encoded_state_size == decoder_cell.state_size, str(decoder_cell.state_size) + str(encoded_state_size)
        logger.debug('encoded_state.get_shape() %s', encoded_state.get_shape())

        assert c.word_embed_size == c.col_emb_size, 'We are docoding one of entity.property from DB or word'
        num_decoder_symbols = total_input_vocab_size
        logger.debug('num_decoder_symbols %s', num_decoder_symbols)

        def decoder(feed_previous_bool, scope='att_decoder'):
            reuse = None if feed_previous_bool else True
            logger.debug('Since our decoder_inputs are in fact targets we feed targets without EOS')
            with tf.variable_scope(scope, reuse=reuse):
                decoder_inputs = targets[:-1]
                if c.use_db_encoder:
                    outputs, state = word_db_embed_attention_decoder(col_embeddings, word_embeddings, c,
                        decoder_inputs, encoded_state, attention_states,
                        self.vocabs_cum_start_idx_low, self.vocabs_cum_start_idx_up,
                        decoder_cell, num_heads=1,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=True,
                        initial_state_attention=c.initial_state_attention)
                else:
                    outputs, state = embedding_attention_decoder(
                        decoder_inputs, encoded_state, attention_states, decoder_cell,
                        num_decoder_symbols, c.word_embed_size, num_heads=1,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=True,
                        initial_state_attention=c.initial_state_attention)
                return outputs + [state]

        *dec_logitss, _dec_state = control_flow_ops.cond(self.feed_previous,
                                                        lambda: decoder(True),
                                                        lambda: decoder(False))
        return targets, target_mask, dec_logitss

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

        with tf.variable_scope('loss_and_eval'), elapsed_timer() as loss_timer:
            self.loss = tf.nn.seq2seq.sequence_loss(dec_logitss, targets[1:], target_mask, softmax_loss_function=None)

            logger.debug('Compliling helper functions which identifies properties type {word, col1, col2,..} of decoded and target words')
            self.dec_vocab_idss_op = tf_trg_word2vocab_id(self.dec_outputs, self.vocabs_cum_start_idx_low, self.vocabs_cum_start_idx_up)
            self.trg_vocab_idss_op = tf_trg_word2vocab_id(targets, self.vocabs_cum_start_idx_low, self.vocabs_cum_start_idx_up)

            def bleu_all():
                '''computed from values stored at model dictionary after eval step'''
                return np.mean(get_bleus(self.trg_utts, self.dec_utts))

            def bleu_words():
                '''computed from values stored at model dictionary after eval step'''
                # words have ids == c.num_cols
                just_wordss_dec = [[w for w, i in zip(utt, ids) if i == c.num_cols] for utt, ids in zip(self.dec_utts, self.dec_vocab_idss)]
                just_wordss_trg = [[w for w, i in zip(utt, ids) if i == c.num_cols] for utt, ids in zip(self.trg_utts, self.trg_vocab_idss)]
                return np.mean(get_bleus(just_wordss_trg, just_wordss_dec))

            def properties_match():
                '''computed from values stored at model dictionary after eval step'''
                # words are stored as last_column see dstc2.target_vocabs
                utts_rewards = [np.mean([1 if dec_vid == trg_vid else 0 for dec_vid, trg_vid in zip(dec_vocab_ids, trg_vocab_ids)])
                        for dec_vocab_ids, trg_vocab_ids in zip(self.dec_vocab_idss, self.trg_vocab_idss)]
                return np.mean(utts_rewards)

            self.eval_functions = [bleu_all, bleu_words, properties_match]
            assert len(self.eval_functions) == len(c.eval_func_weights), str(len(self.eval_functions) == len(c.evla_func_weights))

            logger.debug('Building the loss function ops took %.2f s', loss_timer())

        with tf.variable_scope('updates'), elapsed_timer() as updates_timer:
            # TODO load reward and implement mixer
            self._optimizer = opt = tf.train.AdamOptimizer(c.learning_rate)
            tf.scalar_summary(self.loss.op.name + 'loss', self.loss)
            params = tf.trainable_variables()
            gradients = opt.compute_gradients(self.loss, params)
            # TODO visualize gradients on inputs, rows or whereever needed
            modified_grads = gradients
            self.xent_updates = opt.apply_gradients(modified_grads)
            logger.debug('Building the gradient udpate ops took %.2f s', updates_timer())

            if c.reinforce_first_step >= 0:
                logger.info('Reinforce algorithm will be used after step %d. Compiling reward loading into TF', c.reinforce_first_step)
                logger.debug('Reward need to be computed outside of TF')
                tf.reward = tf.placeholder(tf.float32, shape=(c.batch_size,), name='reward')

                self.expected_reward = tf.get_variable('expected_reward', initializer=tf.zeros([c.batch_size, c.encoder_size], dtype=tf.float32), 
                        trainable=False)
                self.update_expected_reward = assign_moving_average(self.expected_reward, tf.reward, c.reward_moving_avg_decay)
                # raise NotImplementedError('Mixer gradients') # FIXME

        self.summarize = tf.merge_all_summaries()

        times = [inpt_timer(), db_timer(), dec_timer(), loss_timer()]
        logger.debug('Blocks times: %s,\n total: %.2f', times, sum(times))

    def _xent_update(self, sess, train_dict, log_output):
        logger.info('Xent updates for step %7d', self.step)
        if log_output:
            info_dic = self.eval_step(sess, train_dict, log_output)
        else:
            info_dic = {}
        sess.run(self.xent_updates, train_dict)
        return info_dic

    def train_step(self, sess, train_dict, log_output=False):
        self.step += 1
        c = self.config
        if c.reinforce_first_step < self.step:
            self._xent_update(sess, train_dict, log_output)

        logger.info('Rein updates for step %7d', self.step)

        info_dir = self.eval_step(sess, train_dict, train_dict, log_output)
        train_dict['reward'] = info_dir['reward']

        logger.debug('Deciding for each word how to mix RL or Xent updates')
        for i, w_plc in enumerate(reversed(self.mixer_weights_plc)):
            train_dict[w_plc] = sigmoid(self.step - c.reinforce_first_step)

        sess.run([self.mixer_optimizer], feed_dict=train_dict)
        return info_dir

    def eval_step(self, sess, eval_dict, log_output=False):
        c = self.config
        if log_output:
            output_feed = self.dec_outputs + self.dec_vocab_idss_op + self.trg_vocab_idss_op + [self.loss, self.summarize]
        else:
            output_feed = self.dec_outputs + self.dec_vocab_idss_op + self.trg_vocab_idss_op + [self.loss]

        out_vals = sess.run(output_feed, eval_dict)
        mtl = c.max_turn_len
        if log_output:
            loss_v, sum_v = out_vals[-2], out_vals[-1]
            decoder_outs, dec_v_ids, trg_v_ids = out_vals[0:mtl], out_vals[mtl:2 * mtl], out_vals[2 * mtl: 3 * mtl]
        else:
            loss_v = out_vals[-1]
            decoder_outs, dec_v_ids, trg_v_ids = out_vals[0:mtl], out_vals[mtl:2 * mtl], out_vals[2 * mtl: 3 * mtl]

        l_ds = [trim_decoded(utt, c.EOS_ID) for utt in time2batch(decoder_outs)]
        utt_lens, self.dec_utts = zip(*l_ds)
        self.dec_vocab_idss = [ids[:k] for k, ids in zip(utt_lens, time2batch(dec_v_ids))]
        trg_lens = eval_dict['target_lens:0']
        self.trg_vocab_idss = [ids[:k] for k, ids in zip(trg_lens, time2batch(trg_v_ids))]
        self.trg_utts = [utt[:k] for k, utt in zip(trg_lens, eval_dict['dec_targets:0'].tolist())]

        eval_func_vals = [f() for f in self.eval_functions]
        w_eval_func_vals = [w * v for w, v in zip(c.eval_func_weights, eval_func_vals)]
        reward = sum(w_eval_func_vals)

        ret_dir = {'loss': loss_v, 'reward': reward}
        if log_output:
            func_names=[f.__name__ for f in self.eval_functions]
            eval_f_vals_dict = dict(zip(func_names, eval_func_vals))
            w_eval_f_vals_dict = dict(zip(func_names, w_eval_func_vals))

            total_sum = tf.Summary(value=[tf.Summary.Value(tag='reward', simple_value=reward)])
            sum_func_val = tf.Summary(value=[tf.Summary.Value(tag=n, simple_value=v) for n, v in eval_f_vals_dict.items()])
            sum_wfunc_val = tf.Summary(value=[tf.Summary.Value(tag=n, simple_value=v) for n, v in w_eval_f_vals_dict.items()])
            total_sum.MergeFrom(sum_func_val)
            total_sum.MergeFrom(sum_wfunc_val)
            total_sum.MergeFromString(sum_v)
            ret_dir.update({'summarize': total_sum, 'decoder_outputs': decoder_outs})
            ret_dir.update(eval_f_vals_dict)
            ret_dir.update(w_eval_f_vals_dict)

        return ret_dir 

    def decode_step(self, sess, input_feed_dict):
        decoder_outs = sess.run(self.dec_outputs, input_feed_dict)
        return {'decoder_outputs': decoder_outs}

    def log(self, name, writer, step_inputs, step_outputs, e, dstc2_set=None, labels_dt=None):
        c = self.config
        if 'summarize' in step_outputs:
            writer.add_summary(step_outputs['summarize'], self.step)

        logger.debug('\nStep log %s\nEpoch %d Step %d' % (name, e, self.step))
        for k, v in step_outputs.items():
            if k == 'decoder_outputs' or k == 'summarize':
                continue
            logger.debug('  %s: %s' % (k, v))

        if dstc2_set is not None:
            if 'words:0' in step_inputs and 'turn_len:0' in step_inputs:
                binputs, blens = step_inputs['words:0'], step_inputs['turn_len:0']
                for b, (inp, d) in enumerate(zip(binputs, blens)):
                    logger.info('inp %07d,%02d: %s', self.step, b, ' '.join([dstc2_set.words_vocab.get_w(idx) for idx in inp[:d]]))
            if 'decoder_outputs' in step_outputs:
                dec_outs = [trim_decoded(utt, c.EOS_ID)[1] for utt in time2batch(step_outputs['decoder_outputs'])]
                for bout in dec_outs:
                    logger.info('dec %07d,%02d: %s', self.step, b, ' '.join([dstc2_set.get_target_surface(i)[1] for i in bout]))
            if labels_dt is not None and 'dec_targets:0' in labels_dt and 'target_lens:0' in labels_dt:
                btargets, blens = labels_dt['dec_targets:0'], labels_dt['target_lens:0']
                for b, (targets, d) in enumerate(zip(btargets, blens)):
                    logger.info('trg %07d,%02d: %s', self.step, b, ' '.join([dstc2_set.get_target_surface(i)[1] for i in targets[0:d]]))


class FastComp(E2E_property_decoding):
    '''Dummy class just for debugging training loop - it compiles fast.'''
    def __init__(self, config):
        self._var2save = tf.Variable([1])
        self.step, self.config = 0, config

        self._define_inputs(config)
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
