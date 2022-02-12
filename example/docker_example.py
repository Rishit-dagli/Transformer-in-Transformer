import tensorflow as tf
from tnt import TNT

tnt = TNT(
    image_size=256,  # size of image
    patch_dim=512,  # dimension of patch token
    pixel_dim=24,  # dimension of pixel token
    patch_size=16,  # patch size
    pixel_size=4,  # pixel size
    depth=5,  # depth
    num_classes=1000,  # output number of classes
    attn_dropout=0.1,  # attention dropout
    ff_dropout=0.1,  # feedforward dropout
)

img = tf.random.uniform(shape=[5, 3, 256, 256])
logits = tnt(img)
print(logits.shape)
print("Should be (5, 1000)")
