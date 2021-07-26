from .layers import *


def FPN2D(OUTPUT_CHANNELS=3):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
            downsample2D(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            downsample2D(128, 4),  # (bs, 64, 64, 128)
            downsample2D(256, 4),  # (bs, 32, 32, 256)
            downsample2D(512, 4),  # (bs, 16, 16, 512)

    ]

    down_stack_filters = [
            512,
            512,
            512,
            512

    ]

    up_stack = [
            upsample2D(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample2D(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample2D(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            upsample2D(512, 4),  # (bs, 16, 16, 1024)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    up_stack_filters = reversed(down_stack_filters)

    # Upsampling and establishing the skip connections
    for up, skip, filters in zip(up_stack, skips, up_stack_filters):
        x = up(x)
        x = LateralConnect2D(x, skip, filters)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def FPN3D(initial_filter=64, depth=3, shape=[4, 256, 256, 3], OUTPUT_CHANNELS=3):
    inputs = tf.keras.layers.Input(shape=shape)

    down_stack = [
            downsample3D(64, [3, 4, 4], [3, 2, 2], padding='same', apply_batchnorm=False),  # (bs, 4, 128, 128, 64)
            downsample3D(128, [2, 4, 4], [2, 2, 2], padding='same'),  # (bs, 4, 64, 64, 128)
            downsample3D(256, [1, 4, 4], [1, 2, 2], padding='same')  # (bs, 4, 4, 32, 256)

    ]

    down_stack_filters = [
            256,
            256,
            256,

    ]

    up_stack = [
            upsample3D(256, [3, 4, 4], [2, 2, 2], apply_dropout=True),  # (bs, 2, 2, 1024)
            upsample3D(256, [2, 4, 4], [1, 2, 2], apply_dropout=True),  # (bs, 4, 4, 1024)
            upsample3D(256, [1, 4, 4], [3, 2, 2], apply_dropout=True)  # (bs, 8, 8, 1024)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=[1, 2, 2],
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    up_stack_filters = reversed(down_stack_filters)

    # Upsampling and establishing the skip connections
    for up, skip, filters in zip(up_stack, skips, up_stack_filters):
        x = up(x)
        x = LateralConnect3D(x, skip, filters)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


