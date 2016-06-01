#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, subprocess, os, logging, random, argparse, sys
from datetime import datetime
from timeit import default_timer
from contextlib import contextmanager

import tensorflow as tf
import numpy as np

from e2end.dataset.dstc2 import Dstc2, Dstc2DB


logger = logging.getLogger(__name__)


def sigmoid(x):
    x = np.array(x)
    return np.exp(-np.logaddexp(0, -x))


def update_config(c, d):
    for k, v in d.items():
        setattr(c, k, v)


def load_configs(configs):
    config = {}
    for cn in configs:
        c = json.load(open(cn), 'r')
        config.update(c)
    return config


def time2batch(decoder_outputs):
    bsize = len(decoder_outputs[0])
    bouts = [[] for i in range(bsize)]
    # transpose time x batch -> batch_size
    for tout in decoder_outputs:
        for b in range(bsize):
            bouts[b].append(tout[b])
    return bouts


def trim_decoded(decoded_words, EOS_ID):
    try:
        idx = decoded_words.index(EOS_ID)
        return idx, decoded_words[:idx + 1]
    except ValueError:
        logger.debug('no EOS_ID %d %s', EOS_ID, decoded_words)
        return len(decoded_words), decoded_words


def save_config(c, filename):
    json.dump(vars(c), open(filename, 'w'), indent=4, sort_keys=True)


def git_info():
    head, diff, remote = None, None, None
    try:
        head = subprocess.getoutput('git rev-parse HEAD').strip()
    except subprocess.CalledProcessError:
        pass
    try:
        diff = subprocess.getoutput('git diff --no-color')
    except subprocess.CalledProcessError:
        pass
    try:
        remote = subprocess.getoutput('git remote -v').strip()
    except subprocess.CalledProcessError:
        pass
    git_dict = {'head': head or 'Unknown',
            'diff': diff or 'Unknown',
            'remote': remote or 'Unknown'}
    return git_dict


def setup_logging(filename, console_level=logging.INFO):
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG, filename=filename)
    if isinstance(console_level, str):
        console_level = getattr(logging, console_level)
    console = logging.StreamHandler()
    console.setLevel(console_level)
    logging.getLogger('').addHandler(console)


@contextmanager
def elapsed_timer():
    # FIXME rewrite it to use object with __call__ and continue=True arg so it can measure repated statements
    start = default_timer()
    finished_at = None

    def running_elapser():
        return default_timer() - start

    def finished_elapser():
        return finished_at - start

    elapser = running_elapser
    yield lambda: elapser()
    finished_at, elapser = default_timer(), finished_elapser


def launch_tensorboard(logdir, stdout, stderr=subprocess.STDOUT, port=6006):
    '''Launch tensorboard in separate process'''
    if isinstance(stdout, str):
        stdout = open(stdout, 'w')

    try:
        hostname = subprocess.check_output(['hostname', '-d'], universal_newlines=True).strip()
    except subprocess.CalledProcessError:
        hostname = 'unknown'
    logdir = os.path.abspath(logdir)
    if hostname == 'ufal.hide.ms.mff.cuni.cz':
        cmd = ['ssh', 'shrek.ms.mff.cuni.cz', '/home/oplatek/.local/bin/tensorboard --logdir %s --port %d' % (logdir, port)]
        logger.info('Tensorboard launch on shrek') 
        logger.info('Run "ssh oplatek@shrek.ms.mff.cuni.cz -N -L localhost:%d:localhost:6006"', port)
    else:
        cmd = ['tensorboard', '--logdir', logdir, '--port', str(port)]
    process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    logger.info('\n\nTensorboard launched with logdir: %s and port: %d\nTensorboard PID: %d', logdir, port, process.pid)


def parse_input():
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
    ap.add_argument('--row_targets', action='store_true', default=False)
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
    logger.info('Launched\n\n%s\n' % ' '.join(sys.argv))
    logger.debug('Computed also config values on the fly and merged values from config and command line arguments')
    logger.debug('Overwritten config values from command line and setup logging')

    random.seed(c.seed)
    tf.set_random_seed(c.seed)

    with elapsed_timer() as preprocess_timer:
        db = Dstc2DB(c.db_file)
        train = Dstc2(c.train_file, db, just_db=c.row_targets,
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
        from e2end.model import E2E_property_decoding
        m = E2E_property_decoding(c)
    elif c.model == "FastComp":
        from e2end.model.fast_compilation import FastComp
        m = FastComp(c)
    else:
        raise KeyError('Unknown model')
    logger.info('Model %s compiled and loaded', c.model_name)

    c.model_name = m.__class__.__name__
    save_config(c, c.config_filename)
    logger.info('Settings saved to exp config: %s', c.config_filename)
    return c, m, db, train, dev
