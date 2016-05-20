import unittest
from .evaluation import tf_trg_word2vocab_id
import tensorflow as tf


class Trg2vocabidTest(unittest.TestCase):

    def test_borders(self):
        sess = tf.Session()
        with sess.as_default():
            bidx = tf.constant([0, 1, 2, 3, 4, 5, 101, 102, 408, 409, 483], shape=(11, ))
            down = tf.constant([0, 102, 109, 134, 244, 342, 405, 409], shape=(8,))
            up = tf.constant([102, 109, 134, 244, 342, 405, 409, 484], shape=(8,))
            w_arr = [bidx]
            op_arr = tf_trg_word2vocab_id(w_arr, down, up)
            result = op_arr[0].eval()
            result = result.tolist()
            self.assertListEqual(result, [0, 0, 0, 0, 0, 0, 0, 1, 6, 7, 7])

if __name__ == "__main__":
    unittest.main()
