#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Converts 
'''
import argparse
import json


def convert(val_output, outfile, session_keys=None):
    with open(val_output, 'r') as r, open(outfile, 'w') as w:
        data = json.load(r)
        assert 'sessions' in data
        data['dataset'] = val_output
        data['wall-time'] = 0.0 
        for dialog in data['sessions']:
            for turn in dialog['turns']:
                turn['goal-labels'] = {}
                turn['method-label'] = {}
                turn['requested-slots'] = {}
                turn['goal-labels-joint'] = goal_nb = []
                for hyp in turn['nbest']:
                    outputs = hyp['output']
                    tmp = {"food": outputs[0],
                           "area": outputs[1],
                           "pricerange": outputs[2]}
                    slots_dict = {}
                    for k, v in tmp.items():
                        if v != 'none':
                            slots_dict[k] = tmp[k]
                    goal_nb.append({"score": hyp["score"],
                            "slots": slots_dict})

        if session_keys is not None:
            unsorted_sessions = data['sessions']
            sessions_dict = dict([(session['id'], session) for session in unsorted_sessions])
            data['sessions'] = [sessions_dict[key] for key in session_keys]

        json.dump(data, w, indent=4, separators=(',', ': ')) 


def load_session_ids(flist):
    sessions_ids = []
    with open(flist, 'r') as r:
        for line in r:
            session_id = line.strip().split('/')[1]
            sessions_ids.append(session_id)
    return sessions_ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('validated', help=' ')
    ap.add_argument('session_list', help=' ')
    ap.add_argument('out_dstc', help=' ')
    c = ap.parse_args()
    sessions_ids = load_session_ids(c.session_list)
    convert(c.validated, c.out_dstc, session_keys=sessions_ids)
