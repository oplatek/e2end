#!/usr/bin/env python3
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

    def __init__(self, data, max_dial_len=30, max_goals=5, max_constrain=5): 

        goals = ['goal{}'.format(g) for g in range(Dialog.max_goals)]
        cons = ['cons{}'.format(c) for c in range(Dialog.max_constrain)]
        sys_h = ['sys{:02d}'.format(s) for s in range(Dialog.max_dial_len)]
        usr_h = ['usr{:02d}'.format(u) for u in range(Dialog.max_dial_len)]
        turns_h = [i for tup in zip(sys_h, usr_h) for i in tup]
        role = ['role', 'num_usr_replies', 'num_sys_replies']
        goals_asked = ['asked_goal{}'.format(g) for g in range(Dialog.max_goals)]
        # goals_answered = ['answered_goal{}'.format(g) for g in range(Dialog.max_goals)]  # no counter part in the form
        cons_spec = ['cons_requested{}'.format(c) for c in range(Dialog.max_constrain)]
        other_answers = ['client_reply', 'system_reply', 'finished', 'error_found', 'error_utt']

        self.headers = goals + cons + turns_h + role + goals_asked + cons_spec + other_answers
        self.columns_to_fill = goals_asked + cons_spec + other_answers

        self.df = pandas.DataFrame.from_dict(dict([(h, []) for h in self.headers]))
        add = pandas.DataFrame.from_dict(data) if isinstance(data, dict) else data
        self.df = self.df.append(add)

    @staticmethod
    def from_df(self, df):
        d = Dialog({}) 
        return d

    def to_csv(self, filename, sep=',', **kwargs):
        dfd = self.df[self.headers]
        kwargs['sep'] = sep
        kwargs['na_rep'] = ''
        dfd.to_csv(filename, **kwargs)

    def to_csv_for_CF(self, filename, sep=',', **kwargs):
        print('Crowdflower does not like to be provided with the columns which the workers fill in')
        dfd = self.df[self.headers]
        dfd = dfd.drop(self.columns_to_fill, axis=1)
        kwargs['sep'] = sep
        kwargs['na_rep'] = ''
        dfd.to_csv(filename, **kwargs)

    def append(self, d):
        self.df = self.df.append(d.df, ignore_index=True)
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
        df = self.df
        for i, row in df.iterrows():
            if row['client_reply'] == 'dummy':
                df.set_value(i, 'role', 'usr')  # next
                num_sys_repl = int(row['num_sys_replies'])
                num_sys_repl += 1
                df.set_value(i, 'sys%02d' % num_sys_repl, row['system_reply'])
                df.set_value(i, 'num_sys_replies', num_sys_repl)
            elif row['system_reply'] == 'dummy':
                df.set_value(i, 'role', 'sys')  # next
                num_usr_repl = int(row['num_usr_replies'])
                num_usr_repl += 1
                df.set_value(i, 'usr%02d' % num_usr_repl, row['client_reply'])
                df.set_value(i, 'num_usr_replies', num_usr_repl)
            else:
                raise ValueError('One of system or client reply should contain dummy value')

    @staticmethod
    def load_answers(answers, sep=',', **kwargs):
        kwargs['sep'] = sep
        df = pandas.read_csv(answers, **kwargs) 
        # empty strings instead of NaNs 
        df.fillna('', inplace=True)
        return Dialog(df)

    def filter_answered(self):
        finished_cond = self.df['finished'].map(lambda x: x.strip() == 'true')
        notfinished_cond = self.df['finished'].map(lambda x: x.strip() != 'true')
        finished = self.df[finished_cond]
        self.df = self.df[notfinished_cond]
        return Dialog(finished)

    @staticmethod
    def generate_empty_with_goals(db, num, min_goals=2, max_goals=4, 
            constraint_min=2, constraint_max=4, requestable=None, search_by=None):

        col_vocabs = dict(zip(db.column_names, db.col_vocabs))

        requestable = requestable or ['phone', 'pricerange', 'addr', 'area', 'food', 'postcode', 'name']
        search_by = search_by or ['pricerange', 'area', 'food', 'name']
        slot_map = {'phone': 'phone', 'pricerange': 'price_range', 'addr': 'address', 'area': 'area', 'food': 'food_type', 'postcode': 'postcode', 'name': 'name'}

        dialogs = []
        for k in range(num):
            d = {'num_usr_replies': 0, 'num_sys_replies': 0}
            num_goals = min(random.randint(min_goals, max_goals), Dialog.max_goals)
            goals = random.sample(requestable, num_goals)

            num_cons = min(random.randint(constraint_min, constraint_max), Dialog.max_constrain)
            constrains = random.sample(search_by, num_cons)
            constrains = [c for c in constrains if c not in goals]
            constrains = dict([(c, random.sample(list(col_vocabs[c].words), 1)[0]) for c in constrains])
            # print('constraings', constrains)
            for i, g in enumerate(goals):
                d['goal{}'.format(i)] = [slot_map[g]]
            for i, (k, v) in enumerate(constrains.items()):
                if v == 'UNK':  # FIXME hack how to get rid off unwanted values
                    continue
                k = slot_map[k]
                d['cons{}'.format(i)] = ['%s=%s' % (k, v)]
            d['role'] = 'sys'
            dialogs.append(Dialog(d))
        if dialogs:
            return Dialog(pandas.concat([d.df for d in dialogs], ignore_index=True))
        else:
            return Dialog({})

    def initialize_with_hello(self, usr_prob, sys_utts, usr_utts):
        num_rows = len(self.df.index)
        # import ipdb; ipdb.set_trace()

        sys_utts = sys_utts * (int(num_rows / len(sys_utts)) + 1)
        d = dict(zip(range(num_rows), random.sample(sys_utts, num_rows)))
        self.df['sys00'].fillna(d, inplace=True)

        k = int(num_rows * usr_prob)
        usr_utts = usr_utts * (int(k / len(usr_utts)) + 1)
        idx = [num_rows - i - 1 for i in range(k)]
        d = dict(zip(idx, random.sample(usr_utts, k)))
        self.df['usr00'].fillna(d, inplace=True)

        num_usr_replies = ([0] * (num_rows - k)) + ([1] * k)
        num_sys_replies = [1] * num_rows
        role = ['usr' if u < s else 'sys' for u, s in zip(num_usr_replies, num_sys_replies)]
        other = pandas.DataFrame.from_dict({
            'role': role,
            'num_usr_replies': num_usr_replies,
            'num_sys_replies': num_sys_replies})
        self.df.update(other)


if __name__ == "__main__":
    ap = ArgumentParser(__doc__)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--gen_empty', type=int, default=0)
    ap.add_argument('--hello_usr_prob', type=float, default=0.5)
    ap.add_argument('--db_file', default='../../data/dstc2/data.dstc2.db.json')
    ap.add_argument('--answers', default='')
    ap.add_argument('output_file')
    ap.add_argument('finished')

    c = ap.parse_args()
    random.seed(c.seed)

    db = Dstc2DB(c.db_file)
    d = Dialog.generate_empty_with_goals(db, c.gen_empty)
    d.initialize_with_hello(c.hello_usr_prob, ['Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?'], ['Hi'])
    if c.answers:
        a = Dialog.load_answers(c.answers)
        a.move_reply_to_history()
        f = a.filter_answered()
        if len(f.df.index) > 0:
            f.to_csv(c.finished)
        d.append(a)

    d.to_csv_for_CF(c.output_file)
