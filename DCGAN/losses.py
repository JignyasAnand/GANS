import tensorflow as tf


cross_entropy = tf.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(y_true, y_pred):
    real_loss = cross_entropy(tf.ones_like(y_true), y_true)
    fake_loss = cross_entropy(tf.zeros_like(y_pred), y_pred)
    return real_loss + fake_loss

def generator_loss(y_pred):
    return cross_entropy(tf.ones_like(y_pred), y_pred)