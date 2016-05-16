#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import subprocess
import os
import logging
from contextlib import contextmanager
from timeit import default_timer


logger = logging.getLogger(__name__)


class Config:
    @classmethod
    def load_json(cls, filename):
        json_obj = json.load(open(filename), 'r')
        return cls.from_dict(json_obj)

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            c.k = v
        return d

    def to_dict(self):
        d = dict([(a, getattr(self, a)) for a in dir(self) if not a.startswith('__')])
        stripped_d = {k: v for (k, v) in d.items() if not hasattr(v, '__call__')}
        return stripped_d

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def save(self, filename=None):
        json.dump(self.to_dict(), open(filename, 'w'), indent=4, sort_keys=True)


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


def setup_logging(filename):
    logging.basicConfig(level=logging.DEBUG, filename=filename)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)


class Accumulator:
    def __init__(self, stats_lst):
        pass

    def add(self, values):
        pass

    def aggregate(self):
        pass


@contextmanager
def elapsed_timer():
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

    hostname = subprocess.check_output(['hostname', '-d'], universal_newlines=True).strip()
    logdir = os.path.abspath(logdir)
    if hostname == 'ufal.hide.ms.mff.cuni.cz':
        cmd = ['ssh', 'shrek.ms.mff.cuni.cz', '/home/oplatek/.local/bin/tensorboard --logdir %s --port %d' % (logdir, port)]
        logger.info('Tensorboard launch on shrek') 
        logger.info('Run "ssh oplatek@shrek.ms.mff.cuni.cz -N -L localhost:%d:localhost:6006"', port)
    else:
        cmd = ['tensorboard', '--logdir', logdir, '--port', str(port)]
    process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    logger.info('\n\nTensorboard launched with logdir: %s and port: %d\nTensorboard PID: %d', logdir, port, process.pid)
