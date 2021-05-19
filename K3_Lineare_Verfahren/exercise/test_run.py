import unittest
import numpy as np
from run import mse, closed_form_solution
import logging
import sys

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class MyTest(unittest.TestCase):
    # set up wird vor jedem Test aufgerufen
    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([2, 3, 4])

        self.X = np.array([[1, 1], [1, 2], [1, 3]])
        self.y = np.array([1, 2, 3])

    def test_mse(self):
        self.assertEqual(mse(self.a, self.b), 1)

    def test_closed_form_solution(self):
        expected_omega = np.array([0, 1])
        omega = closed_form_solution(self.X, self.y)

        result = np.allclose(expected_omega, omega)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
