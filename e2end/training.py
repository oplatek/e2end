#!/usr/bin/env python
# -*- coding: utf-8 -*-k
import tensorflow as tf
import heapq
import logging


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
        return max(self._heap) if self._heap else 0.0 

    def reset(self):
        self._heap.clear()
