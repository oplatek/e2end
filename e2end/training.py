#!/usr/bin/env python
# -*- coding: utf-8 -*-k
# FIXME implement batch normalization
import tensorflow as tf
import heapq, random, logging
from e2end.utils import elapsed_timer, shuffle, split
import numpy as np


logger = logging.getLogger(__name__)


class EarlyStopperException(Exception):
    pass


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
        if self._not_improved >= self.not_change_limit:
            raise EarlyStopperException()

    def highest_reward(self):
        ''' -666 is dummy value if there is no model logged'''
        if not self._heap:
            return -666  # dummy value
        else:
            reward = heapq.nlargest(1, self._heap)[0][0]
            return reward

    def reset(self):
        self._heap.clear()


def validate(c, sess, m, dev, e, dev_writer):
    with elapsed_timer() as valid_timer:
        logger.info('Sorting dev set according dialog lens.')
        dialog_idx = [i for _, i in sorted(zip(dev.dial_lens.tolist(), range(len(dev))))]
        val_num = 0
        aggreg_func = dict([(fn, 0.0) for fn in [f.__name__ for w, f in zip(c.eval_func_weights, m.eval_functions) if w != 0] + ['loss', 'reward']])

        for d, idxs in enumerate([dialog_idx[b: b+ c.batch_size] for b in range(0, len(dialog_idx), c.batch_size)]):
            if len(idxs) < c.batch_size:
                pad_len = c.batch_size - len(idxs)
                idxs.extend(random.sample(dialog_idx, pad_len))
                logger.info('last batch not alligned, sampling %d the padding', pad_len)

            logger.info('\nValidating dialog %04d', d)
            for t in range(np.max(dev.dial_lens[idxs])):
                logger.info('Validating example %07d', val_num)
                input_fd = {m.turn_len.name: dev.turn_lens[idxs, t],
                            m.is_first_turn: c.history_prefix or t == 0,
                            m.enc_dropout_keep: 1.0,
                            m.dec_dropout_keep: 1.0,
                            m.feed_previous: True,
                            m.dec_targets.name: dev.turn_targets[idxs, t, :],
                            m.target_lens.name: dev.turn_target_lens[idxs, t], 
                            m.gold_rows: dev.gold_rows[idxs, t, :],
                            m.gold_row_lens: dev.gold_row_lens[idxs, t], }
                for k, feat in enumerate(m.feat_list):
                    if k == 0:
                        assert 'words' in feat.name, feat.name
                        input_fd[feat.name] = dev.dialogs[idxs, t, :]
                    elif k == len(m.feat_list) - 1:
                        assert 'speakerId' in feat.name, feat.name
                        input_fd[feat.name] = dev.word_speakers[idxs, t, :]
                    else:
                        input_fd[feat.name] = dev.word_entities[idxs, t, k - 1, :]

                if val_num % c.dev_sample_every == 0:
                    dev_step_outputs = m.eval_step(sess, input_fd, log_output=True)
                    m.log('dev', dev_writer, input_fd, dev_step_outputs, e, dstc2_set=dev, labels_dt=input_fd)
                else:
                    dev_step_outputs = m.eval_step(sess, input_fd)
                for n in aggreg_func:
                    aggreg_func[n] += dev_step_outputs[n]
                val_num += 1
        for n, v in aggreg_func.items():
            aggreg_func[n] = float(v) / np.sum(dev.dial_lens)  # FIXME sum and divide -> AVERAGE may not be the wanted aggregations ops
        validate_set_measures = ([tf.Summary.Value(tag='valid_' + n, simple_value=v) for n, v in aggreg_func.items()])
        dev_writer.add_summary(tf.Summary(value=validate_set_measures), m.step)
        avg_turn_reward, avg_turn_loss = aggreg_func['reward'], aggreg_func['loss']
        logger.info('Step %7d Dev measure dict: %s', m.step, aggreg_func)
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
        logger.info('Sorting train set according dialog lens.')
        buckets = split([i for _, i in sorted(zip(train.dial_lens.tolist(), range(len(train))))], c.num_buckets)
        for e in range(c.epochs):
            logger.debug('\n\nShuffling only withing buckets: %d', e)
            dialog_idx = [i for bucket in buckets for i in shuffle(bucket)]
            assert len(dialog_idx) == len(train), str((len(list(buckets)), len(dialog_idx)))
            for d, idxs in enumerate([dialog_idx[b: b + c.batch_size] for b in range(0, len(dialog_idx), c.batch_size)]):
                logger.info('\nDialog batch %d', d)
                if len(idxs) < c.batch_size:
                    pad_len = c.batch_size - len(idxs)
                    idxs.extend(random.sample(dialog_idx, pad_len))
                    logger.info('last batch not alligned, sampling %d the padding', pad_len)

                for t in range(np.max(train.dial_lens[idxs])):
                    # *_lens are initialized for zeros -> from zeros zero mask
                    input_fd = {m.turn_len.name: train.turn_lens[idxs, t],
                                m.is_first_turn: c.history_prefix or t == 0,
                                m.enc_dropout_keep: c.enc_dropout_keep,
                                m.dec_dropout_keep: c.dec_dropout_keep,
                                m.feed_previous: False,
                                m.dec_targets.name: train.turn_targets[idxs, t, :],
                                m.target_lens.name: train.turn_target_lens[idxs, t],
                                m.gold_rows: train.gold_rows[idxs, t, :],
                                m.gold_row_lens: train.gold_row_lens[idxs, t],
                                }
                    for k, feat in enumerate(m.feat_list):
                        if k == 0:
                            assert 'words' in feat.name, feat.name
                            input_fd[feat.name] = train.dialogs[idxs, t, :]
                        elif k == len(m.feat_list) - 1:
                            assert 'speakerId' in feat.name, feat.name
                            input_fd[feat.name] = train.word_speakers[idxs, t, :]
                        else:
                            input_fd[feat.name] = train.word_entities[idxs, t, k - 1, :]

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
                        if c.reinforce_first_step >= 0 and last_measure_loss and m.step >= c.reinforce_first_step:
                            logger.info('Resetting early stopping from loss to reward')
                            stopper.saver.save(sess=sess, save_path='%s-XENT-final-%.4f-step-%07d' % (stopper.saver_prefix, dev_avg_turn_loss, m.step))
                            stopper.reset() 
                        last_measure_loss = m.step < c.reinforce_first_step
                        stopper.save_and_check(stopper_reward, m.step, sess)
    except KeyboardInterrupt:
        logger.info('\nTraining interrupted manually\n')
    except EarlyStopperException:
        logger.info('\nTraining not improving\n')
    finally:
        logger.info('Training stopped after %7d steps and %7.2f epochs. See logs for %s', m.step, m.step / len(train), config.train_dir)
        logger.info('Saving current state. Please wait!\nBest model has reward %7.2f form step %7d', stopper.highest_reward(), m.step)
        stopper.saver.save(sess=sess, save_path='%s-FINAL-%.4f-step-%07d' % (stopper.saver_prefix, float(stopper_reward), m.step))
