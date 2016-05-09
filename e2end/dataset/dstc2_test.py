#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import os
from .dstc2 import Dstc2


class Dstc2LoadTest(unittest.TestCase):
    def setUp(self):
        self.d = Dstc2(os.path.join(os.path.dirname(__file__), 'dstc2_test_set.json'))

    def test_load(self):
        self.assertIsInstance(self.d, Dstc2)

    def test_labels(self):
        d = self.d
        self.assertListEqual([d.labels_vocab.get_w(i) for i in d.labels.tolist()], ['none none none', "0 1 1", 'none none none', "0 2 1", "0 2 2"])

    def test_inputs(self):
        d = self.d
        m = d.turn_marker
        dl = d.sys_usr_delim
        gold = ['%s %s %s' % (m[0], dl, m[1]), "system 1 1 w1 w2 w3 %s gold 1 1 w2" % dl, '%s %s %s' % (m[0], dl, m[1]), "system 2 1 w1 %s gold 2 1 w4" % dl, "system 2 2 w5 w6 w7 %s gold 2 2 w1" % dl]
        gold = [g.split() for g in gold]
        self.assertTrue(all([len(t) == len(d.turns[0]) for t in d.turns]))
        for t, g in zip(d.turns, gold):
            dt = [d.words_vocab.get_w(i) for i in t]
            self.assertListEqual(dt[:len(g)], g)
            self.assertListEqual(dt[len(g):], ['UNK'] * (d.max_turn_len - len(g)))


if __name__ == "__main__":
    unittest.main()
