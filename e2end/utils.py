#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, subprocess, os, logging
import numpy as np
from contextlib import contextmanager
from timeit import default_timer


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
