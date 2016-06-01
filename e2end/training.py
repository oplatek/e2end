#!/usr/bin/env python
# -*- coding: utf-8 -*-k
# FIXME implement batch normalization
import tensorflow as tf
import heapq, random, logging
from e2end.utils import elapsed_timer


logger = logging.getLogger(__name__)


class TrainingOps(object):
    def __init__(self, loss, optimizer):
        self.optimizer = optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tf.scalar_summary(loss.op.name + 'loss', loss)
        self.train_op = self.optimizer.minimize(loss, global_step=self.global_step)


class EarlyStopper(object):
    '''Keeping track of n_best highest values in reward'''
    def __init__(self, track_n_best, not_change_limit, saver_prefix):
        self.n_best = track_n_best
        self.not_change_limit = not_change_limit
        self._heap = []
        self._not_improved = 0
        self.saver = tf.train.Saver()
        self.saver_prefix = saver_prefix

    @property
    def rewards_steps_sessions(self):
        '''Returns n_best results sorted from the highest to the smallest.'''
        return reversed([heapq.heappop(self._heap) for i in range(len(self._heap))])

    def save_and_check(self, reward, step, sess):

        def save(reward, step):
            path = self.saver.save(sess, '%s-reward-%.4f-step-%07d' % (self.saver_prefix, reward, step))
            logger.info('Sess: %f saved to %s', reward, path)
            return path

        if len(self._heap) < self.n_best:
            self._not_improved = 0
            path = save(reward, step)
            heapq.heappush(self._heap, (reward, step, path))
        else:
            last_reward = self._heap[0][0]
            if last_reward < reward:
                heapq.heappop(self._heap)
                path = save(reward, step)
                heapq.heappush(self._heap, (reward, step, path))
                self._not_improved = 0
            else:
                logger.info('Not keeping reward %f from step %d', reward, step)
                self._not_improved += 1
        return self._not_improved <= self.not_change_limit

    def highest_reward(self):
        ''' -666 is dummy value if there is no model logged'''
        return max(self._heap) if self._heap else -666

    def reset(self):
        self._heap.clear()


def validate(c, sess, m, dev, e, dev_writer):
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
        avg_turn_reward = float(reward / val_num)
        avg_turn_loss = float(loss / val_num)

        validate_set_measures = tf.Summary(value=[tf.Summary.Value(tag='valid_set_reward', simple_value=avg_turn_reward), tf.Summary.Value(tag='valid_set_loss', simple_value=avg_turn_loss)])
        dev_writer.add_summary(validate_set_measures, m.step)
        logger.info('Step %7d Dev reward: %.4f, loss: %.4f', m.step, avg_turn_reward, avg_turn_loss)
    logger.info('Validation finished after %.2f s', valid_timer())
    return avg_turn_reward, avg_turn_loss


def training(c, sess, m, db, train, dev, config, train_writer, dev_writer):
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
                        dev_avg_turn_reward, dev_avg_turn_loss = validate(c, sess, m, dev, e, dev_writer)
                        stopper_reward = - dev_avg_turn_loss if m.step < c.reinforce_first_step else dev_avg_turn_reward
                        if last_measure_loss and m.step > c.reinforce_first_step:
                            logger.info('Resetting early stopping from loss to reward')
                            stopper.saver.save(sess=sess, save_path='%s-XENT-final-%.4f-step-%07d' % (stopper.saver_prefix, dev_avg_turn_loss, m.step))
                            stopper.reset() 
                        last_measure_loss = m.step < c.reinforce_first_step
                        if not stopper.save_and_check(stopper_reward, m.step, sess):
                            raise RuntimeError('Training not improving on train set')
    finally:
        logger.info('Training stopped after %7d steps and %7.2f epochs. See logs for %s', m.step, m.step / len(train), config.train_dir)
        logger.info('Saving current state. Please wait!\nBest model has reward %7.2f form step %7d', stopper.highest_reward()[0], m.step)
        stopper.saver.save(sess=sess, save_path='%s-FINAL-%.4f-step-%07d' % (stopper.saver_prefix, float(stopper_reward), m.step))
