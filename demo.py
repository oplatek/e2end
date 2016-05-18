#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for end2end dialog training.

"""
import logging, random, os, argparse
from datetime import datetime
import tensorflow as tf
from e2end.utils import update_config, load_configs, save_config, git_info, setup_logging, elapsed_timer, launch_tensorboard
from e2end.training import EarlyStopper
from e2end.dataset.dstc2 import Dstc2, Dstc2DB
import e2end.model


def training(sess, m, db, train, dev, config, train_writer, dev_writer):
    with elapsed_timer() as init_timer:
        tf.initialize_all_variables().run(session=sess)

        stopper = EarlyStopper(c.nbest_models, c.not_change_limit, c.name)
    logger.info('Graph initialized in %.2f s', init_timer())

    logger.info('Load DB data')
    sess.run(m.db_rows.initializer, {m.db_row_initializer: db.table})
    sess.run(m.vocabs_cum_start_idx_low.initializer, {m.vocabs_cum_start_initializer: list(train.word_vocabs_downlimit.values())})
    sess.run(m.vocabs_cum_start_idx_up.initializer, {m.vocabs_cum_start_initializer: list(train.word_vocabs_uplimit.values())})

    try:
        step = 0
        dialog_idx = list(range(len(train)))
        logger.info('training set size: %d', len(dialog_idx))
        for e in range(c.epochs):
            logger.debug('\n\nShuffling indexes for next epoch %d', e)
            random.shuffle(dialog_idx)
            for d, i in enumerate(dialog_idx):
                logger.info('\nDialog %d', d)
                for t in range(train.dial_lens[i]):
                    logger.info('Step %d', step)
                    assert c.batch_size == 1, 'FIXME not doing proper batching'  # FIXME
                    labels_dt = {m.dec_targets.name: train.turn_targets[i:i+1, t, :],
                                 m.target_lens.name: train.turn_target_lens[i:i+1, t], }
                    input_fd = {m.turn_len.name: train.turn_lens[i:i+1, t],
                                m.is_first_turn: t == 0,
                                m.dropout_keep_prob: c.dropout,
                                m.dropout_db_keep_prob: c.db_dropout,
                                m.feed_previous: False,
                                }
                    for k, feat in enumerate(m.feat_list):
                        if k == 0:
                            assert 'words' in feat.name, feat.name
                            input_fd[feat.name] = train.dialogs[i:i+1, t, :]
                        elif k == len(m.feat_list) - 1:
                            assert 'speakerId' in feat.name, feat.name
                            input_fd[feat.name] = train.word_speakers[i:i+1, t, :]
                        else:
                            input_fd[feat.name] = train.word_entities[i:i+1, t, k - 1, :]

                    if step % c.train_sample_every == 0:
                        tr_step_outputs = m.eval_step(sess, input_fd, labels_dt)
                        m.log('train', train_writer, input_fd, tr_step_outputs, e, step, dstc2_set=train, labels_dt=labels_dt)
                    elif step % c.train_loss_every == 0:
                        tr_step_outputs = m.train_step(sess, input_fd, labels_dt, log_output=True)
                        m.log('train', train_writer, input_fd, tr_step_outputs, e, step, dstc2_set=train, labels_dt=labels_dt)
                    else:
                        m.train_step(sess, input_fd, labels_dt, log_output=False)

                    if step % c.validate_every == 0:
                        dev_avg_turn_loss = validate(m, dev, e, step, sess, dev_writer)
                        stopper_reward = - dev_avg_turn_loss
                        if not stopper.save_and_check(stopper_reward, step, sess):
                            raise RuntimeError('Training not improving on train set')  # FIXME validate on dev set

                    step += 1
    finally:
        logger.info('Training stopped after %7d steps and %7.2f epochs. See logs for %s', step, step / len(train), config.train_dir)
        logger.info('Saving current state.\nBest model has reward %7.2f form step %7d is %s' % stopper.highest_reward())
        stopper.saver.save(sess=sess, save_path='%s-FINAL-%.4f-step-%07d' % (stopper.saver_prefix, stopper_reward, step))


def validate(m, dev, e, step, sess, dev_writer):
    with elapsed_timer() as valid_timer:
        dialog_idx = list(range(len(dev)))
        logger.info('Selecting randomly %d from %d for validation', len(dialog_idx), len(dev))
        val_num, loss = 0, 0.0
        for d, i in enumerate(dialog_idx):
            logger.info('\nValidating dialog %04d', d)
            for t in range(train.dial_lens[i]):
                logger.info('Validating example %07d', val_num)
                assert c.batch_size == 1, 'FIXME not doing proper batching'  # FIXME`l
                labels_dt = {m.dec_targets.name: train.turn_targets[i:i+1, t, :],
                             m.target_lens.name: train.turn_target_lens[i:i+1, t], }
                input_fd = {m.turn_len.name: train.turn_lens[i:i+1, t],
                            m.is_first_turn: t == 0,
                            m.dropout_keep_prob: 1.0,
                            m.dropout_db_keep_prob: 1.0,
                            m.feed_previous: True}
                # FIXME m.feed_previous should be True
                for k, feat in enumerate(m.feat_list):
                    if k == 0:
                        assert 'words' in feat.name, feat.name
                        input_fd[feat.name] = train.dialogs[i:i+1, t, :]
                    elif k == len(m.feat_list) - 1:
                        assert 'speakerId' in feat.name, feat.name
                        input_fd[feat.name] = train.word_speakers[i:i+1, t, :]
                    else:
                        input_fd[feat.name] = train.word_entities[i:i+1, t, k - 1, :]

                dev_step_outputs = m.eval_step(sess, input_fd, labels_dt)
                if val_num % c.dev_log_sample_every == 0:
                    m.log('dev', dev_writer, input_fd, dev_step_outputs, e, step, dstc2_set=dev, labels_dt=labels_dt)
                loss += dev_step_outputs['loss']
                val_num += 1
        avg_turn_loss = loss / val_num
        logger.info('Step %7d Dev loss: %.4f', step, avg_turn_loss)
    logger.info('Validation finished after %.2f s', valid_timer())
    return avg_turn_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', nargs='*', default=[])
    parser.add_argument('--exp', default='exp')
    parser.add_argument('--use-db-encoder', action='store_true', default=False)
    parser.add_argument('--train-dir', default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--log-console-level', default="INFO")
    parser.add_argument('--train-file', default='./data/artificial/data.dstc2.example1.json')
    parser.add_argument('--db_file', default='./data/artificial/data.dstc2.orthogonal7rowdb.json')
    parser.add_argument('--dev_file', default='./data/artificial/data.dstc2.example1.json')
    parser.add_argument('--word-embed-size', type=int, default=10)
    parser.add_argument('--epochs', default=20000)
    parser.add_argument('--learning_rate', default=0.0005)
    parser.add_argument('--max_gradient_norm', default=5.0)
    parser.add_argument('--validate_every', default=100)
    parser.add_argument('--train_loss_every', default=1000)
    parser.add_argument('--train_sample_every', default=100)
    parser.add_argument('--dev_log_sample_every', default=10)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--dev_batch_size', default=1)
    parser.add_argument('--embedding_size', default=20)
    parser.add_argument('--dropout', default=1.0)
    parser.add_argument('--db_dropout', default=1.0)
    parser.add_argument('--feat_embed_size', default=2)
    parser.add_argument('--nbest_models', default=3)
    parser.add_argument('--not_change_limit', default=100)  # FIXME Be sure that we compare models from different epochs
    parser.add_argument('--encoder_layers', default=1)
    parser.add_argument('--decoder_layers', default=1)
    parser.add_argument('--sample_unk', default=0)
    parser.add_argument('--encoder_size', default=20)
    parser.add_argument('--fast-comp', action='store_true', default=False)
    parser.add_argument('--initial-state-attention', action='store_false', default=True, help='Used for resuming decoding from previous round, kind of what we are doing here')

    c = parser.parse_args()
    conf_dict = load_configs(c.config)
    conf_dict.update(vars(c))
    update_config(c, conf_dict)

    c.name = 'log/%(u)s-%(n)s/%(u)s%(n)s' % {'u': datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S.%f')[:-3], 'n': c.exp}
    c.train_dir = c.train_dir or c.name + '_traindir'
    c.config_filename = '%s.json' % c.name
    c.words_vocab_file = '%s.vocab.words' % c.name
    c.col_vocab_prefix = '%s.vocab.col.' % c.name
    c.log_name = '%s.log' % c.name
    c.tensorboardlog = c.name + '_tensorboard.log'
    c.col_emb_size = c.word_embed_size
    c.mlp_db_l1_size = 6 * c.col_emb_size + c.encoder_size
    c.mlp_db_embed_l1_size = 6 * 10 * c.col_emb_size

    os.makedirs(os.path.dirname(c.name), exist_ok=True)
    setup_logging(c.log_name, console_level=c.log_console_level)
    logger = logging.getLogger(__name__)
    logger.debug('Computed also config values on the fly and merged values from config and command line arguments')
    logger.debug('Overwritten config values from command line and setup logging')

    random.seed(c.seed)
    tf.set_random_seed(c.seed)

    with elapsed_timer() as preprocess_timer:
        db = Dstc2DB(c.db_file)
        train = Dstc2(c.train_file, db, sample_unk=c.sample_unk, first_n=None)
        dev = Dstc2(c.dev_file, db,
                words_vocab=train.words_vocab,
                max_turn_len=train.max_turn_len,
                max_dial_len=train.max_dial_len,
                max_target_len=train.max_target_len,
                first_n=100)
    logger.info('Data loaded in %.2f s', preprocess_timer())

    logger.info('Saving config and vocabularies')
    c.col_vocab_sizes = [len(vocab) for vocab in db.col_vocabs]
    c.max_turn_len = train.max_turn_len
    c.max_target_len = train.max_target_len
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

    if c.fast_comp:
        m = e2end.model.FastComp(c)
    else:
        m = e2end.model.E2E_property_decoding(c)

    c.model_name = m.__class__.__name__
    save_config(c, c.config_filename)
    logger.info('Settings saved to exp config: %s', c.config_filename)

    logger.info('Model %s compiled and loaded', c.model_name)
    launch_tensorboard(c.train_dir, c.tensorboardlog)

    with elapsed_timer() as sess_timer, tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(c.train_dir + '/train', sess.graph)
        dev_writer = tf.train.SummaryWriter(c.train_dir + '/dev', sess.graph)
        logger.debug('Loading session took %.2f', sess_timer())
        training(sess, m, db, train, dev, c, train_writer, dev_writer)
    # TODO if decode only load the model parameter and test it
