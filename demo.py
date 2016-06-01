#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for end2end dialog training.

"""
import logging, sys
import tensorflow as tf
from e2end.utils import elapsed_timer, launch_tensorboard, parse_input
from e2end.debug import setup_debug_hook
from e2end.training import training, validate 

logger = logging.getLogger(__name__)
setup_debug_hook()


if __name__ == "__main__":
    logger.info('Launched\n\n%s\n' % ' '.join(sys.argv[1:]))
    c, m, db, train, dev = parse_input()
    c.tensorboard and launch_tensorboard(c.train_dir, c.tensorboardlog)

    with elapsed_timer() as sess_timer, tf.Session() as sess:
        if c.validate_to_dir is not None:
            logger.info('Just launching validation and NO training')
            dev_writer = tf.train.SummaryWriter(c.validate_to_dir, graph_def=sess.graph if c.save_graph else None)
            validate(c, sess, m, dev, -666, dev_writer)
        else:
            train_writer = tf.train.SummaryWriter(c.train_dir + '/train', graph_def=sess.graph if c.save_graph else None)
            logger.debug('Set up SummaryWriter took %.2f', sess_timer())
            dev_writer = tf.train.SummaryWriter(c.train_dir + '/dev', sess.graph)
            training(c, sess, m, db, train, dev, c, train_writer, dev_writer)
