#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from .utils import compare_ref
from .dataset import Vocabulary
import numpy as np


class CompareRefTest(unittest.TestCase):

    @unittest.skip("just for printing")
    def test_input(self):
        w = Vocabulary([str(i) for i in range(6)])
        l = Vocabulary([str(i) for i in range(2)])
        inp = np.array([[[w.get_i('1'), w.get_i('2'), w.get_i('3')], [w.get_i('4'), w.get_i('5'), w.get_i('0')]]])
        lab = np.array([[l.get_i('1')], [l.get_i('0')]])
        pred = np.array([[l.get_i('1')], [l.get_i('0')]])
        print(compare_ref(inp, lab, pred, w, l))


if __name__ == "__main__":
    unittest.main()
