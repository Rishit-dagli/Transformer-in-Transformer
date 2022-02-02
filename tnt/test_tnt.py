import random
from itertools import permutations

import numpy as np
import pytest
import tensorflow as tf
from parameterized import parameterized

from .tnt import TNT


class TNTTest(tf.test.TestCase):
    def setUp(self):
        super(TNTTest, self).setUp()

    def generate_params():
        param_list = []
        image_size = []
        depth = []
        num_classes = []
        attn_dropout = []
        ff_dropout = []
        batch_size = []
        # not feasible to run more tests due to compute time
        for _ in range(1):
            image_size.append(
                random.randint(1, 64) * 16
            )  # Should be divisible by patch_size
            depth.append(random.randint(1, 10))
            num_classes.append(random.randint(1, 150))
            attn_dropout.append(random.uniform(0, 1))
            ff_dropout.append(random.uniform(0, 1))
            batch_size.append(random.randint(1, 10))
        param_list = [
            [a, b, c, d, e, f]
            for a in image_size
            for b in depth
            for c in num_classes
            for d in attn_dropout
            for e in ff_dropout
            for f in batch_size
        ]
        return param_list

    @parameterized.expand(generate_params())
    def test_shape_and_rank(
        self, image_size, depth, num_classes, attn_dropout, ff_dropout, batch_size
    ):
        tnt = TNT(
            image_size=image_size,
            patch_dim=512,
            pixel_dim=24,
            patch_size=16,
            pixel_size=4,
            depth=depth,
            num_classes=num_classes,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
        # Some values here are harcoded to get away from checking divisibility while generation

        x = tf.random.uniform((batch_size, 3, image_size, image_size))
        y = tnt(x)

        self.assertEqual(tf.rank(y), 2)
        self.assertShapeEqual(np.zeros((batch_size, num_classes)), y)


if __name__ == "__main__":
    tf.test.main()
