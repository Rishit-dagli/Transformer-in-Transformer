import random
from itertools import permutations

import numpy as np
import pytest
import tensorflow as tf
from parameterized import parameterized

from .attention import Attention


class AttentionTest(tf.test.TestCase):
    def setUp(self):
        super(AttentionTest, self).setUp()

    def generate_params():
        param_list = []
        dims = []
        heads = []
        dim_head = []
        dropout = []
        x1 = []
        x2 = []
        x3 = []
        for _ in range(2):
            dims.append(random.randint(1, 1000))
            heads.append(random.randint(1, 10))
            dim_head.append(random.randint(1, 100))
            dropout.append(random.uniform(0, 1))
            x1.append(random.randint(1, 10))
            x2.append(random.randint(1, 512))
            x3.append(random.randint(1, 512))
        param_list = [
            [a, b, c, d, e, f, g]
            for a in dims
            for b in heads
            for c in dim_head
            for d in dropout
            for e in x1
            for f in x2
            for g in x3
        ]
        param_list = random.sample(
            param_list, 16
        )  # because we dont want to run 128 tests
        return param_list

    @parameterized.expand(generate_params())
    def test_shape_and_rank(self, dims, heads, dim_head, dropout, x1, x2, x3):
        attention = Attention(dims, heads, dim_head, dropout)
        x = tf.random.uniform((x1, x2, x3))
        y = attention(x)

        self.assertEqual(tf.rank(y), 3)
        self.assertShapeEqual(np.zeros((x1, x2, dims)), y)


if __name__ == "__main__":
    tf.test.main()
