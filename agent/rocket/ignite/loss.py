import tensorflow as tf  # Deep Learning library


def cross_entropy_loss(model, states, actions, rewards, training=False):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.model(states, training=training),
                                                                  labels=actions) * -rewards)
