import tensorflow as tf
from einops import rearrange, reduce


class Attention(tf.keras.layers.Layer):
    """
    Attention layer.
    This class is much like what I wrote in
    https://github.com/Rishit-dagli/Fast-Transformer/blob/main/fast_transformer/fast_attention.py#L6
    of course not using the additive attention aspect.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()

        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = tf.keras.layers.Dense(
            inner_dim * 3, input_dim=dim, use_bias=False
        )

        self.to_out = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim),
                tf.keras.layers.Dropout(dropout),
            ]
        )

    def call(self, x, **kwargs):
        if len(x.shape) == 4:
            x = x[:, 0, :, :]
        b, n, d = x.shape

        qkv = self.to_qkv(x)
        queries, keys, values = tf.split(qkv, num_or_size_splits=3, axis=-1)
        queries, keys, values = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.heads),
            (queries, keys, values),
        )

        sim = tf.einsum("b i d, b j d -> b i j", queries, keys) * self.scale
        attn = tf.nn.softmax(sim, axis=-1)

        out = tf.einsum("b i j, b j d -> b i d", attn, values)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)

        return self.to_out(out)
