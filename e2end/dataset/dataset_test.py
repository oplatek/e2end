import unittest
from . import Vocabulary


class VocabularyTest(unittest.TestCase):

    def test_input(self):
        idx = list(range(11))
        ws = [str(i) for i in idx]
        v = Vocabulary(ws)

        tidx = [v.get_i(v.get_w(i)) for i in idx]
        self.assertListEqual(idx, tidx)
        tws = [v.get_w(v.get_i(w)) for w in ws]
        self.assertListEqual(ws, tws)

    def test_oov(self):
        v = Vocabulary([])
        self.assertEqual(v.get_i(v.unk), v.get_i('w'))

if __name__ == "__main__":
    unittest.main()
