#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import numpy as np
import tensorflow as tf

import logging, math
from tensorflow.python.ops import control_flow_ops
from .utils import elapsed_timer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class E2E_property_decoding():

    def _define_inputs(self, c):
        self.db_row_initializer = tf.placeholder(tf.int64, shape=(c.num_rows, c.num_cols), name='row_initializer')
        self.db_rows = tf.Variable(self.db_row_initializer, trainable=False, collections=[], name='db_rows')

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
        self.target_lens = tf.placeholder(tf.int64, shape=(c.batch_size,), name='decoder_lengths')

    def __init__(self, config):
        c = config  # shortcut as it is used heavily
        logger.info('Compiling %s', self.__class__.__name__)

        self._define_inputs(c)

        logger.debug('For each word_i from i in 1..max_turn_len there is a list of features: word, belongs2slot1, belongs2slot2, ..., belongs2slotK')
        logger.debug('Feature list uses placelhoder.name to create feed dictionary')

        esingle_cell = tf.nn.rnn_cell.GRUCell(c.encoder_size)
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [esingle_cell] * c.encoder_layers) if c.encoder_layers > 1 else esingle_cell

        logger.debug('The decoder uses special token GO_ID as first input. Adding to vocabulary.')
        self.GO_ID = c.num_words
        self.goid = tf.constant(self.GO_ID)
        goid_batch_vec = tf.constant([self.GO_ID] * c.batch_size, shape=(c.batch_size, 1), dtype=tf.int64)
        logger.debug('Adding GO_ID at the beggining of each decoder input')
        decoder_inputs2D = [goid_batch_vec] + tf.split(1, c.max_target_len, self.dec_targets)
        decoder_inputs = [tf.squeeze(di, [1]) for di in decoder_inputs2D]
        logger.debug('decoder_inputs[0:1].get_shape(): %s, %s', decoder_inputs[0].get_shape(), decoder_inputs[1].get_shape())

        dsingle_cell = tf.nn.rnn_cell.GRUCell(c.num_rows + c.encoder_size + c.encoder_size)
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
            [dsingle_cell] * c.decoder_layers) if c.decoder_layers > 1 else dsingle_cell
        target_mask = [tf.squeeze(m, [1]) for m in tf.split(1, c.max_target_len, lengths2mask2d(self.target_lens, c.max_target_len))]

        with tf.variable_scope('encoder'), elapsed_timer() as inpt_timer:
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

            embedded_inputs = []
            for j in range(c.max_turn_len):
                features_j_word = []
                for i in range(len(self.feat_list)):
                    embedded = tf.nn.embedding_lookup(feat_embeddings[i], self.feat_list[i][:, j])
                    dropped_embedded = tf.nn.dropout(embedded, self.dropout_keep_prob)
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
            logger.debug('Initialization of encoder inputs and control flow took  %.2f s.', inpt_timer())
            words_hidden_feat, dialog_state_after_turn = tf.nn.rnn(encoder_cell, embedded_inputs,
                                                                   initial_state=dialog_state_before_turn,
                                                                   sequence_length=self.turn_len)
            dialog_state_before_acc = tf.assign(dialog_state_before_acc, dialog_state_after_turn)

        with tf.variable_scope('db_encoder'), elapsed_timer() as db_timer:
            col_embeddings = [tf.get_variable('col_values_embedding{}'.format(i),
                                              initializer=tf.random_uniform([col_vocab_size, c.col_emb_size],
                                                                            -math.sqrt(3), math.sqrt(3))) for
                              i, col_vocab_size in enumerate(c.col_vocab_sizes)]

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
            logger.debug('Create embeddings of  all db rows took %.2f s', db_timer())

            batched_db_rows_embeddings = [tf.tile(tf.expand_dims(re, 0), [c.batch_size, 1]) for re in
                                          db_rows_embeddings]

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

        with tf.variable_scope('decoder'), elapsed_timer() as dec_timer:
            # Take from tf/python/ops/seq2seq.py:706
            # First calculate a concatenation of encoder outputs to put attention on.
            words_and_db = words_hidden_feat + [db_embed]
            logger.debug('top_states[0].get_shape() %s', words_hidden_feat[0].get_shape())
            top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size])
                          for e in words_and_db]
            attention_states = tf.concat(1, top_states)
            logger.debug('attention_states.get_shape() %s', attention_states.get_shape())

            total_input_vocab_size = sum(c.col_vocab_sizes + [c.num_words])
            decoder_cell = tf.nn.rnn_cell.OutputProjectionWrapper(decoder_cell, total_input_vocab_size)

            encoded_state = tf.concat(1, [dialog_state_after_turn, tf.squeeze(row_selected, [2]), db_embed])
            encoded_state_size = encoded_state.get_shape().as_list()[1]
            assert encoded_state_size == decoder_cell.state_size, str(decoder_cell.state_size) + str(encoded_state_size)
            logger.debug('encoded_state.get_shape() %s', encoded_state.get_shape())

            assert c.word_embed_size == c.col_emb_size, 'We are docoding one of entity.property from DB or word'
            num_decoder_symbols = total_input_vocab_size
            logger.debug('num_decoder_symbols %s', num_decoder_symbols)

            # If feed_previous is a Tensor, we construct 2 graphs and use cond.

            def decoder(feed_previous_bool, scope='att_decoder'):
                reuse = None if feed_previous_bool else True
                with tf.variable_scope(scope, reuse=reuse):
                    outputs, state = embedding_attention_decoder(
                        decoder_inputs, encoded_state, attention_states, decoder_cell,
                        num_decoder_symbols, c.word_embed_size, num_heads=1,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=True,
                        initial_state_attention=False)
                    return outputs + [state]

            *dec_outputs, dec_state = control_flow_ops.cond(self.feed_previous,
                                                            lambda: decoder(True),
                                                            lambda: decoder(False))
            self.dec_outputs = dec_outputs
            logger.debug('Building of the decoder took %.2f s.', dec_timer())

        with tf.variable_scope('loss'), elapsed_timer() as loss_timer:
            # TODO load reward and implement mixer
            logger.debug('decoder_inputs are targets shifted by one')
            self.loss = tf.nn.seq2seq.sequence_loss(dec_outputs[1:], decoder_inputs[1:], target_mask, softmax_loss_function=None)
            self._optimizer = opt = tf.train.AdamOptimizer(c.learning_rate)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.scalar_summary(self.loss.op.name + 'loss', self.loss)
            params = tf.trainable_variables()
            gradients = opt.compute_gradients(self.loss, params)

            # TODO visualize gradients on inputs, rows or whereever needed
            modified_grads = gradients

            self.updates = opt.apply_gradients(modified_grads, global_step=self.global_step)
            logger.debug('Building the loss function and gradient udpate ops took %.2f s', loss_timer())

        self.summarize = tf.merge_all_summaries()

        times = [inpt_timer(), db_timer(), dec_timer(), loss_timer()]
        logger.debug('Blocks times: %s,\n total: %.2f', times, sum(times))

    def train_step(self, session, input_feed_dict, labels_dict, log_output=False):
        train_dict = input_feed_dict.copy()
        train_dict.update(labels_dict)
        if log_output:
            updates_v, loss_v, sum_v = session.run([self.updates, self.loss, self.summarize], train_dict)
            return {'loss': loss_v, 'summarize': sum_v}
        else:
            session.run(self.updates, train_dict)
            return {}

    def decode_step(self, session, input_feed_dict):
        return session.run(self.dec_outputs, input_feed_dict)

    def eval_step(self, session, input_feed_dict, labels_dict):
        output_feed = self.dec_outputs + [self.loss, self.summarize]
        eval_dict = input_feed_dict.copy()
        eval_dict.update(labels_dict)
        targets, target_lens = labels_dict[self.dec_targets], labels_dict[self.target_lens]

        *decoder_outs, loss_v, sum_v = session.run(output_feed, eval_dict)
        reward = self.evaluate(decoder_outs, targets, target_lens)
        return {'decoder_outputs': decoder_outs, 'loss': loss_v, 'summarize': sum_v, 'reward': reward}

    def evaluate(self, outputs, targets, target_lens):
        return -666

    def log(self, name, writer, step_outputs, e, step, dstc2_set=None):
        if 'summarize' in step_outputs:
            writer.add_summary(step_outputs['summarize'], e)

        logger.debug('\nStep log %s\nEpoch %d Step %d' % (name, e, step))
        for k, v in step_outputs.items():
            if k == 'decoder_outputs' or k == 'summarize':
                continue
            logger.debug('  %s: %s' % (k, v))

        if dstc2_set is not None and 'decoder_outputs' in step_outputs:
            out = ' '.join([dstc2_set.get_target_surface(i) for i in step_outputs['decoder_outputs']])
            logger.debug(out)


def embedding_attention_decoder(decoder_inputs, initial_state, attention_states,
                                cell, num_symbols, embedding_size, num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=tf.float32, scope=None,
                                initial_state_attention=False):
    """RNN decoder with embedding and attention and a pure-decoding option.

    Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

    Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
    ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = tf.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    def _extract_argmax_and_embed(embedding, output_projection=None,
                                  update_embedding=True):
        """Get a loop_function that extracts the previous symbol and embeds it.

        Args:
          embedding: embedding tensor for symbols.
          output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
          update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

        Returns:
          A loop function.
        """
        def loop_function(prev, _):
            if output_projection is not None:
                prev = tf.nn.xw_plus_b(
                    prev, output_projection[0], output_projection[1])
            prev_symbol = tf.argmax(prev, 1)
            # Note that gradients will not propagate through the second parameter of
            # embedding_lookup.
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            if not update_embedding:
                emb_prev = tf.stop_gradient(emb_prev)
            return emb_prev
        return loop_function

    with tf.variable_scope(scope or "embedding_attention_decoder"):
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [num_symbols, embedding_size])

        # TODO replace _extract_argmax_and_embed with sample and embed ideally multiple times
        # implement switch use_inputs, feed_previous, sample
        loop_function = _extract_argmax_and_embed(embedding, output_projection, update_embedding_for_previous) \
                            if feed_previous else None
    emb_inp = [
        tf.nn.embedding_lookup(embedding, i) for i in decoder_inputs]
    # FIXME how to use atten_length > 1 internally it uses conv2d, how to use it
    return tf.nn.seq2seq.attention_decoder(
        emb_inp, initial_state, attention_states, cell, output_size=output_size,
        num_heads=num_heads, loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def lengths2mask2d(lengths, max_len):
    batch_size = lengths.get_shape().as_list()[0]
    # Filled the row of 2d array with lengths
    lengths_transposed = tf.expand_dims(lengths, 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, max_len])

    # Filled the rows of 2d array 0, 1, 2,..., max_len ranges 
    rng = tf.to_int64(tf.range(0, max_len, 1))
    range_row = tf.expand_dims(rng, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])

    # Use the logical operations to create a mask
    return tf.to_float(tf.less(range_tiled, lengths_tiled))


class FastComp(E2E_property_decoding):
    '''Dummy class just for debugging training loop - it compiles fast.'''
    def __init__(self, config):
        self._var2save = tf.Variable([1])

        self._define_inputs(config)
        arr = [
                  self.turn_len, self.dec_targets, self.target_lens,
                  self.is_first_turn, self.feed_previous,
                  self.dropout_keep_prob, self.dropout_db_keep_prob,
              ] + self.feat_list
        self.testTrainOp = tf.concat(0, [tf.to_float(tf.reshape(x, (-1, 1))) for x in arr])

    def train_step(self, session, input_feed_dict, labels_dict, log_output=False):
        train_dict = input_feed_dict.copy()
        train_dict.update(labels_dict)
        session.run(self.testTrainOp, train_dict)
        print('input_feed_dict', input_feed_dict)
        print('input_feed_dict_shape', [(k, v.shape) if hasattr(v, 'shape') else (k, v) for k, v in input_feed_dict.items()])
        print('\nlabels_dict', labels_dict)
        print('labels_dict', [(k, v.shape) if hasattr(v, 'shape') else (k, v) for k, v in labels_dict.items()])
        return

    def decode_step(self, session, input_feed_dict):
        print('input_feed_dict', input_feed_dict)
        return [0, 0, 0]

    def eval_step(self, session, input_feed_dict, labels_dict):
        return (666, [0, 0, 0], -666)

    def log(self, name, writer, step_outputs, e, step):
        pass
