import unittest
import numpy as np
from run import mse
import logging
import sys

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class MyTest(unittest.TestCase):

    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([2, 3, 4])

    def test_mse(self):
        self.assertEqual(mse(self.a, self.b), 1)

    def test_closed_form_solution(self):
        pass


if __name__ == '__main__':
    unittest.main()
