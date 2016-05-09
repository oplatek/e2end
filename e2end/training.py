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
    '''Keeping track of n_best highest values in measure'''
    def __init__(self, track_n_best, not_change_limit, saver_prefix):
        self.n_best = track_n_best
        self.not_change_limit = not_change_limit
        self._heap = []
        self._not_improved = 0
        self.saver = tf.train.Saver()
        self.saver_prefix = saver_prefix

    @property
    def measures_steps_sessions(self):
        '''Returns n_best results sorted from the highest to the smallest.'''
        return reversed([heapq.heappop(self._heap) for i in range(len(self._heap))])

    def save_and_check(self, measure, step, sess):
        heapq.heappush(self._heap, (measure, step, sess))
        logger.debug('New measure %f from step %d', measure, step)
        if len(self._heap) <= self.n_best:
            self._not_improved = 0
        else:
            pop_measure, pop_step, _ = heapq.heappop(self._heap)
            if pop_measure == measure and pop_step ==step:
                logger.info('Not keeping measure %f from step %d', measure, step)
                self._not_improved += 1
            else:
                self._not_improved = 0
        if self._not_improved == 0:  # we stored the model
            path = self.saver.save(sess, '%s-%.4f-%7d' % (self.saver_prefix, measure, step))
            logger.info('Sess: %f saved to %s', measure, path)
        return self._not_improved <= self.not_change_limit
