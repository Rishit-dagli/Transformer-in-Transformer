import tensorflow as tf
from einops import rearrange


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, dim, mult=4, dropout=0.0):
        super(FeedForward, self).__init__()

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim * mult, input_dim=dim),
                tf.keras.layers.Lambda(
                    lambda x: x * tf.math.sqrt(0.5)
                ),  # Gelu activation
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dim, input_dim=dim * mult),
            ]
        )

    def call(self, x):
        return self.net(x)
