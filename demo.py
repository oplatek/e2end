#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for end2end dialog training.

"""
import logging, sys
import tensorflow as tf
from e2end.utils import elapsed_timer, launch_tensorboard, parse_input
from e2end.debug import setup_debug_hook
from e2end.training import training, validate, load_db_data

logger = logging.getLogger(__name__)
setup_debug_hook()


if __name__ == "__main__":
    logger.info('Launched\n\n%s\n' % ' '.join(sys.argv[1:]))
    c, m, db, train, dev = parse_input()
    not c.tensorboard or launch_tensorboard(c.train_dir, c.tensorboardlog)

    if c.cluster:
        logger.warning('Reducing parallel opts for cluster! May degrade performance for GPU!')
        config = tf.ConfigProto(inter_op_parallelism_threads=4,
                        intra_op_parallelism_threads=4) 
    else:
        config = tf.ConfigProto()
    with elapsed_timer() as sess_timer, tf.Session(config=config) as sess:
        if c.validate_to_dir is not None:
            logger.warning('\n\nJust launching validation and NO training\n\n')
            assert c.load_model is not None, 'You should load the trained model'
            assert len(c.config) > 0, 'You should have set the default parameter to the training parameters by loading training config but config list is empty: %s' % str(c.config)
            logger.warning('Change the dev_file to the file for validation')

            s = tf.train.Saver()
            s.restore(sess, c.load_model)
            logger.info('Trainable model parameters restored')
            load_db_data(sess, m, train, db)  
            logger.info('Initialized  model static parameters about DB')
            dev_writer = tf.train.SummaryWriter(c.validate_to_dir, graph_def=sess.graph if c.save_graph else None)
            validate(c, sess, m, dev, -666, dev_writer)
        else:
            train_writer = tf.train.SummaryWriter(c.train_dir + '/train', graph_def=sess.graph if c.save_graph else None)
            logger.debug('Set up SummaryWriter took %.2f', sess_timer())
            dev_writer = tf.train.SummaryWriter(c.train_dir + '/dev', sess.graph)
            training(c, sess, m, db, train, dev, c, train_writer, dev_writer)
