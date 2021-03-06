#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from collections import Counter
import random


__all__ = ['dstc.Dstc2', 'dstc.Dstc2DB']

unk_label = 'UNK'


class Vocabulary:

    @classmethod
    def load_json(cls, filename):
        d = json.load(open(filename, 'r'))
        return cls(**d)

    def __init__(self, counts, max_items=5000, extra_words=None, unk=unk_label):
        self._counts = counts if isinstance(counts, Counter) else Counter(counts)
        self.unk = unk = unk or []
        extra_words = extra_words or []
        self.extra_words = set(extra_words)
        if unk:
            self.extra_words.add(unk)
        self.max_items = max_items
        tmp = sorted(list(set(list(self.extra_words) + [w for w, _ in self._counts.most_common(max_items)])))
        self._w2int = dict(((w, i) for i, w in enumerate(tmp)))
        self._int2w = dict(((i, w) for i, w in enumerate(tmp)))
        self._unk_id = self._w2int[unk] if unk else -666

    @property
    def words(self):
        return iter(self._w2int.keys())

    def indexes(self):
        return iter(self._int2w.keys())

    def get_i(self, w, unk_chance_smaller=0):
        if unk_chance_smaller:
            wc = self._counts[w]
            if wc <= unk_chance_smaller and wc > random.uniform(0, unk_chance_smaller):
                return self._w2int[self.unk]
            else:
                return self._w2int.get(w, self._unk_id)
        else:
            res = self._w2int.get(w, self._unk_id)
            return res

    def get_w(self, index):
        return self._int2w[index]

    def __repr__(self):
        dct = {'unk': self.unk, 'max_items': self.max_items, 'int2w': self._int2w}
        return str(dct)

    def todict(self):
        return {'counts': self._counts,
                'max_items': self.max_items,
                'unk': self.unk,
                'extra_words': list(self.extra_words)}

    def save(self, filename):
        json.dump(self.todict(), open(filename, 'w'), indent=4, sort_keys=True)

    def __len__(self):
        return len(self._w2int)
