import tensorflow as tf


def get_discriminator():
    discriminator = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 4, padding='same', input_shape=(28, 28, 1)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, 4, padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])

    return discriminator


def get_generator():
    generator = tf.keras.models.Sequential([
        tf.keras.layers.Dense(7*7*256, input_shape=(100, ), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Reshape((7,7,256)),

        tf.keras.layers.Conv2DTranspose(128, 5, padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(64, 5, strides=(2, 2), padding="same", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Conv2DTranspose(1, 5, strides=(2, 2), padding="same", use_bias=False, activation="tanh"),
    ])
    return generator