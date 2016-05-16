#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import subprocess
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


def launch_tensorboard(logdir, stdout, stderr=subprocess.STDOUT):
    '''Launch tensorboard in separate process'''
    port = 6006
    if isinstance(stdout, str):
        stdout = open(stdout, 'w')
    process = subprocess.Popen(['tensorboard', '--logdir', logdir, '--port', str(port)], stdout=stdout, stderr=stderr)
    logger.info('\n\nTensorboard launched with logdir: %s and port: %d\nTensorboard PID: %d', logdir, port, process.pid)
    # FIXME detect if using UFAL infrastructure launch ssh and launch tensorfboard on shrek
    # and print out how to ssh tunel to shrek ssh oplatek@shrek.ms.mff.cuni.cz -N -L localhost:6006:localhost:6006
