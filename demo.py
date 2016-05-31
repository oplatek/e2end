#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for end2end dialog training.

"""
import logging, random, os, argparse, sys
from datetime import datetime
import tensorflow as tf
from e2end.utils import update_config, load_configs, save_config, git_info, setup_logging, elapsed_timer, launch_tensorboard
from e2end.debug import setup_debug_hook
from e2end.training import EarlyStopper
from e2end.dataset.dstc2 import Dstc2, Dstc2DB
import e2end.model
import e2end.model.fast_compilation


setup_debug_hook()


def training(sess, m, db, train, dev, config, train_writer, dev_writer):
    with elapsed_timer() as init_timer:
        tf.initialize_all_variables().run(session=sess)
        logger.info('Graph initialized in %.2f s', init_timer())

    with elapsed_timer() as load_db_data:
        sess.run(m.db_rows.initializer, {m.db_row_initializer: db.table})
        sess.run(m.vocabs_cum_start_idx_low.initializer, {m.vocabs_cum_start_initializer: list(train.word_vocabs_downlimit.values())})
        sess.run(m.vocabs_cum_start_idx_up.initializer, {m.vocabs_cum_start_initializer: list(train.word_vocabs_uplimit.values())})
        logger.info('DB data loaded in %0.2f s', load_db_data())

    stopper, stopper_reward, last_measure_loss = EarlyStopper(c.nbest_models, c.not_change_limit, c.name), 0.0, True
    tf.get_default_graph().finalize()
    try:
        dialog_idx = list(range(len(train)))
        logger.info('training set size: %d', len(dialog_idx))
        for e in range(c.epochs):
            logger.debug('\n\nShuffling indexes for next epoch %d', e)
            random.shuffle(dialog_idx)
            for d, i in enumerate(dialog_idx):
                logger.info('\nDialog %d', d)
                for t in range(train.dial_lens[i]):
                    assert c.batch_size == 1, 'FIXME not doing proper batching'  # FIXME
                    input_fd = {m.turn_len.name: train.turn_lens[i:i+1, t],
                                m.is_first_turn: t == 0,
                                m.dropout_keep_prob: c.dropout,
                                m.dropout_db_keep_prob: c.db_dropout,
                                m.feed_previous: False,
                                m.dec_targets.name: train.turn_targets[i:i+1, t, :],
                                m.target_lens.name: train.turn_target_lens[i:i+1, t],
                                m.gold_rows: train.gold_rows[i:i+1, t, :],
                                m.gold_row_lens: train.gold_row_lens[i:i+1, t],
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

                    m.step_increment()
                    if m.step % c.train_loss_every == 0:
                        tr_step_outputs = m.train_step(sess, input_fd, log_output=True)
                        m.log('train', train_writer, input_fd, tr_step_outputs, e, dstc2_set=train, labels_dt=input_fd)
                    else:
                        m.train_step(sess, input_fd)
                    if m.step % c.train_sample_every == 0:
                        tr_step_outputs = m.eval_step(sess, input_fd, log_output=True)
                        m.log('train', train_writer, input_fd, tr_step_outputs, e, dstc2_set=train, labels_dt=input_fd)

                    if m.step % c.validate_every == 0:
                        dev_avg_turn_reward, dev_avg_turn_loss = validate(sess, m, dev, e, dev_writer)
                        stopper_reward = - dev_avg_turn_loss if m.step < c.reinforce_first_step else dev_avg_turn_reward
                        if last_measure_loss and m.step > c.reinforce_first_step:
                            logger.info('Resetting early stopping from loss to reward')
                            stopper.saver.save(sess=sess, save_path='%s-XENT-final-%.4f-step-%07d' % (stopper.saver_prefix, dev_avg_turn_loss, m.step))
                            stopper.clear() 
                        last_measure_loss = m.step < c.reinforce_first_step
                        if not stopper.save_and_check(stopper_reward, m.step, sess):
                            raise RuntimeError('Training not improving on train set')
    finally:
        logger.info('Training stopped after %7d steps and %7.2f epochs. See logs for %s', m.step, m.step / len(train), config.train_dir)
        logger.info('Saving current state. Please wait!\nBest model has reward %7.2f form step %7d', float(stopper.highest_reward()), m.step)
        stopper.saver.save(sess=sess, save_path='%s-FINAL-%.4f-step-%07d' % (stopper.saver_prefix, float(stopper_reward), m.step))


def validate(sess, m, dev, e, dev_writer):
    with elapsed_timer() as valid_timer:
        dialog_idx = list(range(len(dev)))
        logger.info('Selecting randomly %d from %d for validation', len(dialog_idx), len(dev))
        val_num, reward, loss = 0, 0.0, 0.0
        for d, i in enumerate(dialog_idx):
            logger.info('\nValidating dialog %04d', d)
            for t in range(dev.dial_lens[i]):
                logger.info('Validating example %07d', val_num)
                assert c.batch_size == 1, 'FIXME not doing proper batching'
                input_fd = {m.turn_len.name: dev.turn_lens[i:i+1, t],
                            m.is_first_turn: t == 0,
                            m.dropout_keep_prob: 1.0,
                            m.dropout_db_keep_prob: 1.0,
                            m.feed_previous: True,
                            m.dec_targets.name: dev.turn_targets[i:i+1, t, :],
                            m.target_lens.name: dev.turn_target_lens[i:i+1, t], 
                            m.gold_rows: dev.gold_rows[i:i+1, t, :],
                            m.gold_row_lens: dev.gold_row_lens[i:i+1, t], }
                for k, feat in enumerate(m.feat_list):
                    if k == 0:
                        assert 'words' in feat.name, feat.name
                        input_fd[feat.name] = dev.dialogs[i:i+1, t, :]
                    elif k == len(m.feat_list) - 1:
                        assert 'speakerId' in feat.name, feat.name
                        input_fd[feat.name] = dev.word_speakers[i:i+1, t, :]
                    else:
                        input_fd[feat.name] = dev.word_entities[i:i+1, t, k - 1, :]

                if val_num % c.dev_sample_every == 0:
                    dev_step_outputs = m.eval_step(sess, input_fd, log_output=True)
                    m.log('dev', dev_writer, input_fd, dev_step_outputs, e, dstc2_set=dev, labels_dt=input_fd)
                else:
                    dev_step_outputs = m.eval_step(sess, input_fd)
                reward += dev_step_outputs['reward']
                loss += dev_step_outputs['loss']
                val_num += 1
        avg_turn_reward = reward / val_num
        avg_turn_loss = loss / val_num
        logger.info('Step %7d Dev reward: %.4f, loss: %.4f', m.step, avg_turn_reward, avg_turn_loss)
    logger.info('Validation finished after %.2f s', valid_timer())
    return avg_turn_reward, avg_turn_loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--config', nargs='*', default=[])
    ap.add_argument('--exp', default='exp')
    ap.add_argument('--validate_to_dir', default=None)
    ap.add_argument('--save_graph', action='store_true', default=False)
    ap.add_argument('--tensorboard', action='store_true', default=False)
    ap.add_argument('--train_dir', default=None)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--log_console_level', default="INFO")
    ap.add_argument('--train_file', default='./data/dstc2/data.dstc2.train.json')
    ap.add_argument('--dev_file', default='./data/dstc2/data.dstc2.dev.json')
    ap.add_argument('--db_file', default='./data/dstc2/data.dstc2.db.json')
    ap.add_argument('--train_first_n', type=int, default=None)
    ap.add_argument('--dev_first_n', type=int, default=None)

    ap.add_argument('--model', default='E2E_property_decoding')
    ap.add_argument('--use_db_encoder', action='store_true', default=False)
    ap.add_argument('--just_db', action='store_true', default=False)
    ap.add_argument('--eval_func_weights', type=float, nargs='*', default=[0.0, 0.0, 0.0, 0.5, 0.5], help='''
            If row accuracy and row coverage has weights 0.5 and 0.5 then its sum is row F1 score. 
            We should slightly prefer coverage, especially at the beggining of training.''')

    ap.add_argument('--encoder_size', type=int, default=12)
    ap.add_argument('--word_embed_size', type=int, default=11)
    ap.add_argument('--encoder_layers', type=int, default=1)
    ap.add_argument('--decoder_layers', type=int, default=1)
    ap.add_argument('--max_gradient_norm', type=float, default=5.0)
    ap.add_argument('--reward_moving_avg_decay', type=float, default=0.99)
    ap.add_argument('--dropout', type=float, default=1.0)
    ap.add_argument('--db_dropout', type=float, default=1.0)
    ap.add_argument('--feat_embed_size', type=int, default=2)
    ap.add_argument('--initial_state_attention', action='store_false', default=True, help='Used for resuming decoding from previous round, kind of what we are doing here')
    ap.add_argument('--learning_rate', type=float, default=0.0005)
    ap.add_argument('--mixer_learning_rate', type=float, default=0.0005)

    ap.add_argument('--reinforce_first_step', type=int, default=sys.maxsize)
    ap.add_argument('--reinforce_next_step', type=int, default=5000)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--train_sample_every', type=int, default=100)
    ap.add_argument('--train_loss_every', type=int, default=100)
    ap.add_argument('--validate_every', type=int, default=500)
    ap.add_argument('--nbest_models', type=int, default=3)
    ap.add_argument('--not_change_limit', type=int, default=100)  # FIXME Be sure that we compare models from different epochs
    ap.add_argument('--sample_unk', type=int, default=0)
    ap.add_argument('--dev_sample_every', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--dev_batch_size', type=int, default=1)

    c = ap.parse_args()
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
    if c.validate_to_dir is not None:
        c.log_name = os.path.join(c.validate_to_dir, os.path.basename(c.log_name))
    setup_logging(c.log_name, console_level=c.log_console_level)
    logger = logging.getLogger(__name__)
    logger.debug('Computed also config values on the fly and merged values from config and command line arguments')
    logger.debug('Overwritten config values from command line and setup logging')

    random.seed(c.seed)
    tf.set_random_seed(c.seed)

    with elapsed_timer() as preprocess_timer:
        db = Dstc2DB(c.db_file)
        train = Dstc2(c.train_file, db, just_db=c.just_db,
                      sample_unk=c.sample_unk, first_n=c.train_first_n)
        dev = Dstc2(c.dev_file, db,
                just_db=train.just_db,
                words_vocab=train.words_vocab,
                max_turn_len=train.max_turn_len,
                max_dial_len=train.max_dial_len,
                max_target_len=train.max_target_len,
                max_row_len=train.max_row_len,
                first_n=c.dev_first_n)
    logger.info('Data loaded in %.2f s', preprocess_timer())

    logger.info('Saving config and vocabularies')
    c.EOS_ID = int(train.get_target_surface_id('words', train.words_vocab, train.EOS))
    c.col_vocab_sizes = [len(vocab) for vocab in db.col_vocabs]
    c.max_turn_len = train.max_turn_len
    c.max_target_len = train.max_target_len
    c.max_row_len = train.max_row_len
    c.column_names = db.column_names
    c.num_words = len(train.words_vocab)
    c.num_rows = db.num_rows
    c.num_cols = db.num_cols
    c.restaurant_name_vocab_id = db.get_col_idx('name')
    c.name_low, c.name_up = int(train.word_vocabs_downlimit['name']), int(train.word_vocabs_uplimit['name'])
    c.git_info = git_info()
    logger.info('Config\n\n: %s\n\n', c)
    logger.info('Saving helper files')
    train.words_vocab.save(c.words_vocab_file)
    for vocab, name in zip(db.col_vocabs, db.column_names):
        vocab.save(c.col_vocab_prefix + name)

    if c.model == "E2E_property_decoding":
        m = e2end.model.E2E_property_decoding(c)
    elif c.model == "FastComp":
        m = e2end.model.fast_compilation.FastComp(c)
    else:
        raise KeyError('Unknown model')

    c.model_name = m.__class__.__name__
    save_config(c, c.config_filename)
    logger.info('Settings saved to exp config: %s', c.config_filename)

    logger.info('Model %s compiled and loaded', c.model_name)
    c.tensorboard and launch_tensorboard(c.train_dir, c.tensorboardlog)

    with elapsed_timer() as sess_timer, tf.Session() as sess:
        train_writer = tf.train.SummaryWriter(c.train_dir + '/train', graph_def=sess.graph if c.save_graph else None)
        logger.debug('Setting up SummaryWriter took %.2f', sess_timer())
        if c.validate_to_dir is not None:
            logger.info('Just launching validation and NO training')
            dev_writer = tf.train.SummaryWriter(c.validate_to_dir, graph_def=sess.graph if c.save_graph else None)
            validate(sess, m, dev, -666, dev_writer)
        else:
            dev_writer = tf.train.SummaryWriter(c.train_dir + '/dev', sess.graph)
            training(sess, m, db, train, dev, c, train_writer, dev_writer)
