#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import os
from .dstc2 import Dstc2, Dstc2DB


class Dstc2LoadTest(unittest.TestCase):
    def setUp(self):
        db = Dstc2DB(os.path.join(os.path.dirname(__file__), 'dstc2_db_test.json'))
        self.d = Dstc2(os.path.join(os.path.dirname(__file__), 'dstc2_test_set.json'), db)

    def test_load(self):
        self.assertIsInstance(self.d, Dstc2)

    # FIXME
    # def test_inputs(self):
    #     d = self.d
    #     gold = [["system 1 1 w1 w2 w3 gold 1 1 w2"], ["system 2 1 w1 gold 2 1 w4", "system 2 2 w5 w6 w7 gold 2 2 w1"]]
    #     gold = [[t.split() for t in g] for g in gold]
    #     for dia, gdia in zip(d.dialogs, gold):
    #         dt = [[d.words_vocab.get_w(i) for i in t] for t in dia]
    #         print(dt)
    #         self.assertListEqual(dt[:len(g)], g)
    #         self.assertListEqual(dt[len(g):], ['UNK'] * (d.max_turn_len - len(g)))


if __name__ == "__main__":
    unittest.main()
