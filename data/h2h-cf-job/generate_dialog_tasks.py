#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate continue dialogue Crowdflower task with goal for DSTC2 db.
"""
if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    from e2end.dataset.dstc2 import Dstc2DB

from argparse import ArgumentParser
import random
import pandas
import numpy as np

from e2end.debug import setup_debug_hook


setup_debug_hook()


class Dialog:
    max_dial_len = 30
    max_goals = 5
    max_constrain = 5

    def __init__(self, data): 
        self.df = pandas.DataFrame.from_dict(dict([(h, []) for h in self.get_headers()]))
        add = pandas.DataFrame.from_dict(data) if isinstance(data, dict) else data
        self.df = self.df.append(add)

    @staticmethod
    def from_df(self, df):
        d = Dialog({}) 
        return d

    def to_csv(self, filename, sep='\t', **kwargs):
        kwargs['sep'] = sep
        kwargs['cols'] = self.get_headers()
        kwargs['header'] = True
        kwargs['na_rep'] = ''
        self.df.to_csv(filename, **kwargs)

    def merge(self, d):
        self.df.merge(d.df)
        return self

    def shuffle(self, inplace=True, axis=0):     
        self.df.apply(np.random.shuffle, axis=axis)
        return self

    def move_reply_to_history(self, validate=False, check=True):
        '''
        For each row
            Find out role (if len(sys) == len(usr) -> sys reply else usr reply
            Find the first column name sys$i or usr$i which is empty for i
            move the reply to that column
        '''

        pass

    @staticmethod
    def get_headers():

        goals = ['goal{}'.format(g) for g in range(Dialog.max_goals)]
        cons = ['cons{}'.format(c) for c in range(Dialog.max_constrain)]
        sys_h = ['sys{:02d}'.format(s) for s in range(Dialog.max_dial_len)]
        usr_h = ['usr{:02d}'.format(u) for u in range(Dialog.max_dial_len)]
        error = ['error{:02d}'.format(e) for e in range(Dialog.max_dial_len)]
        turns_h = [i for tup in zip(sys_h, usr_h, error) for i in tup]
        task = ['task']
        goals_asked = ['asked_goal{}'.format(g) for g in range(Dialog.max_goals)]
        goals_answered = ['answered_goal{}'.format(g) for g in range(Dialog.max_goals)]
        cons_spec = ['specified_constrain{}'.format(c) for c in range(Dialog.max_constrain)]
        author = ['author']

        return goals + cons + turns_h + task + goals_asked + goals_answered + cons_spec + author + error

    @staticmethod
    def load_answers(answers, sep='\t', **kwargs):
        kwargs['sep'] = sep
        df = pandas.read_csv(answers, **kwargs) 
        # empty strings instead of NaNs 
        df.fillna('', inplace=True)
        return Dialog(df)

    @staticmethod
    def filter_answered():
        '''TODO check rows if dialog finished'''
        pass   

    @staticmethod
    def generate_empty_with_goals(db, num, min_goals=2, max_goals=4, 
            constraint_min=1, constraint_max=2, requestable=None, search_by=None):
        col_vocabs = dict(zip(db.column_names, db.col_vocabs))

        requestable = requestable or ['phone', 'pricerange', 'addr', 'area', 'food', 'postcode', 'name']
        search_by = search_by or ['pricerange', 'area', 'food', 'name']

        dialogs = []
        for k in range(num):
            d = {}
            num_goals = min(random.randint(min_goals, max_goals), max_goals)
            goals = random.sample(requestable, num_goals)

            num_cons = min(random.randint(constraint_min, constraint_max), Dialog.max_constrain)
            constrains = random.sample(search_by, num_cons)
            constrains = [c for c in constrains if c not in goals]
            constrains = dict([(c, random.sample(list(col_vocabs[c].words), 1)[0]) for c in constrains])
            for i, g in enumerate(goals):
                d['goal{}'.format(i)] = [g]
            for i, c in enumerate(constrains):
                d['cons{}'.format(i)] = [c]
            dialogs.append(Dialog(d))

            return Dialog(pandas.concat([d.df for d in dialogs]))
        else:
            return Dialog({})


if __name__ == "__main__":
    ap = ArgumentParser(__doc__)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--gen_empty', type=int, default=0)
    ap.add_argument('--db_file', default='../../data/dstc2/data.dstc2.db.json')
    ap.add_argument('--answers', default='')
    ap.add_argument('output_file')

    c = ap.parse_args()
    random.seed(c.seed)

    db = Dstc2DB(c.db_file)
    d = Dialog.generate_empty_with_goals(db, c.gen_empty)
    if c.answers:
        a = Dialog.load_answers(c.answers)
        a.filter_answered()
        d = d.merge(a)

    d.to_csv(c.output_file)
