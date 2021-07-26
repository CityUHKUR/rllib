import tensorflow as tf
import numpy as np


def conv_kernel_initializer(shape,
                            dtype=tf.dtypes.float32,
                            partition_info=None):
    """Initialization for convolutional kernels.
    The main difference with tf.variance_scaling_initializer is that
    tf.variance_scaling_initializer uses a truncated normal with an uncorrected
    standard deviation, whereas here we use a normal distribution. Similarly,
    tf.initializers.variance_scaling uses a truncated normal with
    a corrected standard deviation.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)
    return tf.random_normal_initializer(mean=0.0,
                                        stddev=np.sqrt(2.0 / fan_out))(shape=shape,
                                                                       dtype=dtype)


def dense_kernel_initializer(shape,
                             dtype=tf.dtypes.float32,
                             partition_info=None):
    """Initialization for dense kernels.
    This initialization is equal to
      tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                      distribution='uniform').
    It is written out explicitly here for clarity.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    init_range = 1.0 / np.sqrt(shape[1])
    return tf.random_uniform_initializer(-init_range,
                                         init_range)(shape=shape,
                                                     dtype=dtype)


def superpixel_kernel_initializer(shape,
                                  dtype=tf.dtypes.float32,
                                  partition_info=None):
    """Initializes superpixel kernels.
    This is inspired by space-to-depth transformation that is mathematically
    equivalent before and after the transformation. But we do the space-to-depth
    via a convolution. Moreover, we make the layer trainable instead of direct
    transform, we can initialization it this way so that the model can learn not
    to do anything but keep it mathematically equivalent, when improving
    performance.
    Args:
      shape: shape of variable
      dtype: dtype of variable
      partition_info: unused
    Returns:
      an initialization for the variable
    """
    del partition_info
    #  use input depth to make superpixel kernel.
    depth = shape[-2]
    filters = np.zeros([2, 2, depth, 4 * depth], dtype=dtype)
    i = np.arange(2)
    j = np.arange(2)
    k = np.arange(depth)
    mesh = np.array(np.meshgrid(i, j, k)).T.reshape(-1, 3).T
    filters[
        mesh[0],
        mesh[1],
        mesh[2],
        4 * mesh[2] + 2 * mesh[0] + mesh[1]] = 1
    return filters
