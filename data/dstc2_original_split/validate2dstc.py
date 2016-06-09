#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Converts 
'''
import argparse
import json


def convert(val_output, outfile):
    with open(val_output, 'r') as r, open(outfile, 'w') as w:
        data = json.load(r)
        assert 'sessions' in data
        data['dataset'] = val_output
        for i, dialog in enumerate(data['sessions']):
            dialog['id'] = i
            for turn in dialog['turns']:
                turn['goal-labels'] = {}
                turn['goal-labels-joint'] = goal_nb = []
                for hyp in turn['nbest']:
                    outputs = hyp['output']
                    slots_dict = {
                            "food": outputs[0],
                            "area": outputs[1],
                            "pricerange": outputs[2]}
                    for k, v in slots_dict.items():
                        if v == 'none':
                            del k
                    goal_nb.append({"score": hyp["score"],
                            "slots": slots_dict})
        json.dump(data, w, sort_keys=True, indent=4, separators=(',', ': ')) 


if __name__ == "__main__":
    ap = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('validated', help=' ')
    ap.add_argument('out_dstc', help=' ')
    c = ap.parse_args()
    convert(c.validated, c.out_dstc)
