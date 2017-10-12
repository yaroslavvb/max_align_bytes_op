import tensorflow as tf
import numpy as np

import unittest

class Test(unittest.TestCase):
  def test_simple(self):
    sess = tf.Session()
    self.assertEqual(sess.run(tf.ones(())*2), np.ones(())*2)
