import tensorflow as tf
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange
from .attention import Attention


def get_elements_from_nested_list(l, new_l):
    if l is not None:
        e = l[0]
        if isinstance(e, list):
            get_elements_from_nested_list(e, new_l)
        else:
            new_l.append(e)
        if len(l) > 1:
            return get_elements_from_nested_list(l[1:], new_l)
        else:
            return new_l


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
                tf.keras.layers.Dense(dim * mult),
                tf.keras.layers.Activation(tf.nn.gelu),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(dim),
            ]
        )

    def call(self, x):
        return self.net(x)


class TNT(tf.keras.layers.Layer):
    def __init__(
        self,
        image_size,
        patch_dim,
        pixel_dim,
        patch_size,
        pixel_size,
        depth,
        num_classes,
        heads=8,
        dim_head=64,
        ff_dropout=0.0,
        attn_dropout=0.0,
        unfold_args=None,
        **kwargs,
    ):
        super(TNT, self).__init__(**kwargs)

        if image_size % patch_size != 0:
            raise ValueError(
                f"Image size must be divisible by patch size: {image_size} / {patch_size}"
            )
        if patch_size % pixel_size != 0:
            raise ValueError(
                f"Patch size must be divisible by pixel size: {patch_size} / {pixel_size}"
            )

        self.image_size = image_size
        self.patch_size = patch_size

        num_patch_tokens = (image_size // patch_size) ** 2
        self.patch_tokens = tf.Variable(
            tf.random.uniform(shape=(num_patch_tokens + 1, patch_dim))
        )

        if unfold_args is None:
            unfold_args = (pixel_size, pixel_size, 0)
        if len(unfold_args) == 2:
            unfold_args = (*unfold_args, 0)
        kernel_size, stride, padding = unfold_args

        pixel_width = int(((patch_size - kernel_size + (2 * padding)) / stride) + 1)
        num_pixels = pixel_width ** 2

        self.to_pixel_tokens = tf.keras.Sequential(
            [
                Rearrange(
                    "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=patch_size, p2=patch_size
                ),
                Rearrange("... c n -> ... n c"),
                tf.keras.layers.Dense(pixel_dim, input_dim=3 * kernel_size ** 2),
            ]
        )

        self.patch_pos_emb = tf.Variable(
            tf.random.uniform(shape=(num_patch_tokens + 1, patch_dim))
        )
        self.pixel_pos_emb = tf.Variable(
            tf.random.uniform(shape=(num_pixels, pixel_dim))
        )

        layers = []
        for _ in range(depth):
            pixel_to_patch = [
                tf.keras.layers.LayerNormalization(axis=-1),
                Rearrange("... n d -> ... (n d)"),
                tf.keras.layers.Dense(patch_dim),
            ]

            layers.append(
                [
                    PreNorm(
                        pixel_dim,
                        Attention(
                            dim=pixel_dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                    )
                ]
            )

            layers.append(
                PreNorm(pixel_dim, FeedForward(dim=pixel_dim, dropout=ff_dropout))
            )

            layers.append(pixel_to_patch)

            layers.append(
                [
                    PreNorm(
                        patch_dim,
                        Attention(
                            dim=patch_dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=attn_dropout,
                        ),
                    )
                ]
            )

            layers.append(
                PreNorm(patch_dim, FeedForward(dim=patch_dim, dropout=ff_dropout))
            )
        self.layers = layers

        self.mlp_head = tf.keras.Sequential(
            [
                tf.keras.layers.LayerNormalization(axis=-1),
                tf.keras.layers.Dense(num_classes, input_dim=patch_dim),
            ]
        )

    def call(self, x):
        batches, channels, height, width = x.shape.as_list()
        patch_size, image_size = self.patch_size, self.image_size

        if height % patch_size != 0:
            raise ValueError(
                f"Image height must be divisible by patch size: {height} / {patch_size}"
            )
        if width % patch_size != 0:
            raise ValueError(
                f"Image width must be divisible by patch size: {width} / {patch_size}"
            )

        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        n = num_patches_w * num_patches_h

        pixels = self.to_pixel_tokens(x)
        patches = repeat(self.patch_tokens[: (n + 1)], "n d -> b n d", b=batches)

        patches += rearrange(self.patch_pos_emb[: (n + 1)], "n d -> () n d")
        rearrange(self.pixel_pos_emb, "n d -> () n d")
        pixels += rearrange(self.pixel_pos_emb, "n d -> () n d")

        layer_group = 0
        pixel_attn, pixel_ff, pixel_to_patch_residual, patch_attn, patch_ff = (
            [],
            [],
            [],
            [],
            [],
        )
        for layer in self.layers:
            if layer_group % 5 == 0:
                pixel_attn.append(layer)
            if layer_group % 5 == 1:
                pixel_ff.append(layer)
            if layer_group % 5 == 2:
                pixel_to_patch_residual.append(layer)
            if layer_group % 5 == 3:
                patch_attn.append(layer)
            if layer_group % 5 == 4:
                patch_ff.append(layer)
            layer_group += 1

        if len(pixels.shape) > 3:
            pixels = pixels[:, 0, :, :]

        for i, j in zip(pixel_attn, pixel_ff):
            pixels = (
                tf.keras.Sequential(i)(pixels) + tf.keras.Sequential(j)(pixels) + pixels
            )

        pixel_to_patch_residual_layers = get_elements_from_nested_list(
            pixel_to_patch_residual, []
        )

        for i, j in zip(patch_attn, patch_ff):
            patches = (
                tf.keras.Sequential(i)(patches)
                + tf.keras.Sequential(j)(patches)
                + patches
            )

        cls_token = patches[:, 0]
        return self.mlp_head(cls_token)
