#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for dialogue state tracking.

"""
import logging, uuid, random, os, argparse
import tensorflow as tf
from e2end.utils import Config, git_info, setup_logging, Accumulator, elapsed_timer
from e2end.training import EarlyStopper
from e2end.dataset.dstc2 import Dstc2, Dstc2DB
import e2end.model


def decode(m, dev, step, sess, i, t, dev_writer):
    with elapsed_timer() as valid_timer:
        acc_log = Accumulator(['mean', 'discard', 'activations', 'discard'])
        for di in range(dev):
            for dt in range(dev.dial_lens[i]):
                feed_dct = {m.dropout_keep_prob: 1.0,
                        m.new_turn: dev.dialogs[i, t, :],
                        m.turn_len: dev.turn_lens[i, t],
                        m.is_first_turn: 1 if t == 0 else 0,
                        m.attention_mask: dev.att_mask[i, t, :],
                        m.turn_targets: dev.turn_targets[i, t, :],
                        m.turn_target_len: dev._turn_target_lens[i, t]}
            log_vals = sess.run([m.loss, m.predictions, m.att_time_db, m.summarize], 
                    feed_dict=feed_dct)
            logger.debug(m.log('dev', dev_writer, log_vals))
            acc_log.add('dev', log_vals)
        logger.info('Decoding val set finished after %.2f s', valid_timer())
        agg_log_vals = acc_log.aggregate()
        dev_acc = agg_log_vals['acc']
        logger.info('Step %7d Dev Accuracy: %.4f', step, dev_acc)
    logger.info('Validation finished in %.2f s.', valid_timer())
    return dev_acc


def training(sess, m, train, dev, config, train_writer, dev_writer):
    with elapsed_timer() as init_timer:
        logger.info('Monitor progress by tensorboard:\ntensorboard --logdir "%s"\n', c.train_dir)
        tf.initialize_all_variables().run(session=sess)

        stopper = EarlyStopper(c.nbest_models, c.not_change_limit, c.name)
    logger.info('Graph initialized in %.2f s', init_timer())

    try:
        step = 0
        for i in random.shuffle(range(len(train))):
            for t in range(train.dial_lens[i]):
                feed_dct = {m.dropout_keep_prob: c.dropout,
                        m.new_turn: train.dialogs[i, t, :],
                        m.turn_len: train.turn_lens[i, t],
                        m.is_first_turn: 1 if t == 0 else 0,
                        m.attention_mask: train.att_mask[i, t, :],
                        m.turn_targets: train.turn_targets[i, t, :],
                        m.turn_target_len: train._turn_target_lens[i, t]}
                if step % c.train_sample_every == 0:
                    _, step, *log_vals = sess.run(
                            [m.train_op, m.global_step, m.loss, m.predictions, 
                             m.att_time_db, m.summarize], feed_dict=feed_dct)
                    logger.debug(m.log('train', train_writer, log_vals))
                else:
                    _, step = sess.run([t.train_op, t.global_step], feed_dict=feed_dct)

                if step % c.validate_every == 0:
                    dev_acc = decode(m, dev, step, sess, i, t, dev_writer)
                if not stopper.save_and_check(dev_acc, step, sess):
                    break
    finally:
        stopper.saver.save('%s-FINAL-STOP-%7d' % (stopper.saver_prefix, step))
        logger.info('Training stopped after %7d steps and %7.2f epochs', step, step / len(train))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--name', default='log/%(u)s/%(u)s' % {'u': uuid.uuid1()})
    args = parser.parse_args()

    c = Config()  
    c.name=args.name
    c.train_dir = c.name + '_traindir'
    c.filename = '%s.json' % c.name
    c.words_vocab_file = '%s.vocab.words' % c.name
    c.col_vocab_prefix = '%s.vocab.col.' % c.name
    c.log_name = '%s.log' % c.name
    c.seed=123
    c.train_file = './data/dstc2/data.dstc2.train.json'
    c.dev_file = './data/dstc2/data.dstc2.dev.json'
    c.db_file = './data/dstc2/data.dstc2.db.json'
    c.epochs = 2
    c.sys_usr_delim = ' SYS_USR_DELIM '
    c.learning_rate = 0.00005
    c.max_gradient_norm = 5.0
    c.validate_every = 200
    c.train_sample_every = 200
    c.batch_size = 2
    c.dev_batch_size = 1
    c.embedding_size=200
    c.dropout = 1.0
    c.rnn_size = 600
    c.feat_embed_size = 2
    c.nbest_models=3
    c.not_change_limit = 5  # FIXME Be sure that we compare models from different epochs
    c.encoder_layers = 1
    c.decoder_layers = 1
    c.sample_unk = 0
    c.encoder_size = 100
    c.word_embed_size = c.col_emb_size = 10
    c.mlp_db_l1_size = 6 * c.col_emb_size + c.encoder_size
    c.mlp_db_embed_l1_size = 6 * 10 * c.col_emb_size
    c.model_name='E2E_property_decoding'
    c.model_name='FastComp'

    os.makedirs(os.path.dirname(c.name), exist_ok=True)
    setup_logging(c.log_name)
    logger = logging.getLogger(__name__)

    random.seed(c.seed)

    db = Dstc2DB(c.db_file)
    train = Dstc2(c.train_file, db, sample_unk=c.sample_unk, first_n=2 * c.batch_size)
    dev = Dstc2(c.dev_file, db,
            words_vocab=train.words_vocab,
            max_turn_len=train.max_turn_len,
            max_dial_len=train.max_dial_len,
            max_target_len=train.max_target_len,
            first_n=2 * c.dev_batch_size)

    logger.info('Saving config and vocabularies')
    c.col_vocab_sizes = [len(vocab) for vocab in db.col_vocabs]
    c.max_turn_len = train.max_turn_len
    c.max_target_len = train.max_turn_len
    c.column_names = db.column_names
    c.num_words = len(train.words_vocab)
    c.num_rows = db.table.shape[0]
    c.num_cols = db.table.shape[1]
    c.git_info = git_info()
    logger.info('Config\n\n: %s\n\n', c)
    logger.info('Saving helper files')
    train.words_vocab.save(c.words_vocab_file)
    for vocab, name in zip(db.col_vocabs, db.column_names):
        vocab.save(c.col_vocab_prefix + name)
    c.save(c.filename)

    m = getattr(e2end.model, c.model_name)(c)
    logger.info('Model %s compiled and loaded', m.__class__.__name__)

    with elapsed_timer() as sess_timer, tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(c.train_dir + '/train', sess.graph)
        dev_writer = tf.train.SummaryWriter(c.train_dir + '/test', sess.graph)
        logger.debug('Loading session took %.2f', sess_timer())
        training(sess, m, train, dev, c, train_writer, dev_writer)

        # TODO if decode only load the model parameter and test it
