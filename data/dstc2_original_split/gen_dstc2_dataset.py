#!/usr/bin/env python3
'''
Generates data for training from oficial formats
'''
import glob
import argparse
import json
import os
import re
from random import shuffle

import wget


def download_dstc2_data():
    if not os.path.isfile('./tmp/dstc2_traindev.tar.gz'):
        wget.download('http://camdial.org/~mh521/dstc/downloads/dstc2_traindev.tar.gz', './tmp')
        wget.download('http://camdial.org/~mh521/dstc/downloads/dstc2_test.tar.gz', './tmp')
        wget.download('http://camdial.org/~mh521/dstc/downloads/dstc2_scripts.tar.gz', './tmp')
        print()
    else:
        print('Everything appears to be downloaded.')


def unpack():
    os.makedirs('./tmp/dstc2_traindev', exist_ok=True)
    os.makedirs('./tmp/dstc2_test', exist_ok=True)
    if not os.path.isdir('./tmp/dstc2_traindev/data'):
        os.system('tar xvzf ./tmp/dstc2_traindev.tar.gz -C ./tmp/dstc2_traindev')
        os.system('tar xvzf ./tmp/dstc2_test.tar.gz -C ./tmp/dstc2_test')
        os.system('tar xvzf ./tmp/dstc2_scripts.tar.gz -C ./tmp')


def gen_slot(slot, goal):
    # if slot in goal:
    #     return slot + '-' + goal[slot].replace(' ', '-')
    # else:
    #     return slot + '-' + 'none'
    if slot in goal:
        return goal[slot]
    else:
        return 'none'


def extract_data(file_name):
    turns = []
    with open(file_name) as flabel:
        with open(file_name.replace('label.json', 'log.json')) as flog:
            label = json.load(flabel)
            log = json.load(flog)

            for lab_turn, log_turn in zip(label['turns'], log['turns']):
                system = log_turn['output']['transcript']
                system = re.sub(r'There are \d+ restaurants', 'There are # restaurants', system)
                system = re.sub(r'There are  restaurants', 'There are # restaurants', system)
                user = lab_turn['transcription']
                user_asr = log_turn['input']['live']['asr-hyps'][0]['asr-hyp']
                user_asr_score = log_turn['input']['live']['asr-hyps'][0]['score']

                state = []
                state.append(gen_slot('food', lab_turn['goal-labels']))
                state.append(gen_slot('area', lab_turn['goal-labels']))
                state.append(gen_slot('pricerange', lab_turn['goal-labels']))
                # state.append(gen_slot('signature', lab_turn['goal-labels']))
                # state.append(gen_slot('postcode', lab_turn['goal-labels']))
                # state.append(gen_slot('addr', lab_turn['goal-labels']))
                # state.append(gen_slot('phone', lab_turn['goal-labels']))
                # state.append(gen_slot('name', lab_turn['goal-labels']))
                state = ' '.join(state)

                print(system)
                print(user)
                print(user_asr)
                print(user_asr_score)
                print(state)
                print('-' * 120)

                turns.append((system, user, user_asr, user_asr_score, state))
    conversation = {'turns': turns, 'session-id': label['session-id']}
    return conversation


def gen_data(dir_name, flist=None):
    conversations = []
    jsons = []
    if flist is None:
        jsons.extend(glob.glob(os.path.join(dir_name, '*/label.json')))
        jsons.extend(glob.glob(os.path.join(dir_name, '*/*/label.json')))
        jsons.extend(glob.glob(os.path.join(dir_name, '*/*/*/label.json')))
    else:
        with open(flist, 'r') as r:
            for line in r:
                json = os.path.join(dir_name, line.strip(), 'label.json')
                assert os.path.isfile(json), json
                jsons.append(json)

    for js in jsons:
        print('Processing a file: {fn}'.format(fn=js))
        conversations.append(extract_data(js))

    return conversations


if __name__ == '__main__':
    ap = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--random_split', action='store_true', default=False, help=' ')
    c = ap.parse_args()
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    download_dstc2_data()
    unpack()

    conversations_train = gen_data('./tmp/dstc2_traindev/data', './tmp/dstc2_traindev/scripts/config/dstc2_train.flist')
    conversations_dev = gen_data('./tmp/dstc2_traindev/data', './tmp/dstc2_traindev/scripts/config/dstc2_dev.flist')
    conversations_test = gen_data('./tmp/dstc2_test/data', './tmp/dstc2_test/scripts/config/dstc2_test.flist')

    # I resplit the data as the train and test are not balanced
    # - there are only 16 'There are' phrases in the original traindev data
    # - but there are 1210 'There are' phrases in the original traindev data
    original = not c.random_split 
    if original:
        with open('./data.dstc2.train.json', 'w') as f:
            json.dump(conversations_train, f, sort_keys=True, indent=4, separators=(',', ': '))
        with open('./data.dstc2.dev.json', 'w') as f:
            json.dump(conversations_dev, f, sort_keys=True, indent=4, separators=(',', ': '))
        with open('./data.dstc2.test.json', 'w') as f:
            json.dump(conversations_test, f, sort_keys=True, indent=4, separators=(',', ': '))
    else:
        conversations = conversations_test + conversations_train + conversations_test
        shuffle(conversations)

        len70 = int(len(conversations) * 0.7)
        len80 = int(len(conversations) * 0.8)
        with open('./data.dstc2.train.json', 'w') as f:
            json.dump(conversations[:len70], f, sort_keys=True, indent=4, separators=(',', ': '))
        with open('./data.dstc2.dev.json', 'w') as f:
            json.dump(conversations[len70:len80], f, sort_keys=True, indent=4, separators=(',', ': '))
        with open('./data.dstc2.test.json', 'w') as f:
            json.dump(conversations[len80:], f, sort_keys=True, indent=4, separators=(',', ': '))
