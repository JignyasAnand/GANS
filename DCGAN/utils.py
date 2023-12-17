import tensorflow as tf
import numpy as np
from config import CONFIG

def get_data(name):
    train_images, train_labels = None, None
    if name == "digit_mnist":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        print("Successfully loaded Digit_MNIST data")
        print("Shapes : ")
        print(f"TRAIN : {train_images.shape}, {test_images.shape}")
        print(f"TEST : {test_images.shape}, {test_labels.shape}")
        print("Now concatenating the datasets")
        train_images = np.concatenate((train_images, test_images))
        train_labels = np.concatenate((train_labels, test_labels))
        print(f"TRAIN : {train_images.shape}, {train_labels.shape}")
    if (train_images is None) or (train_labels is None):
        raise Exception("None type values are being returned")
    if train_images.shape[0] != train_labels.shape[0]:
        raise Exception("Number of train images does not match number of train labels")
    return train_images, train_labels


def preprocess(images, name):
    if name == "digit_mnist":
        print("Preprocessing Digit MNIST data")
        images = np.expand_dims(images, axis=-1)
        print("New Shape : ", images.shape)
        images = (images - 127.5) / 127.5
    return images

def make_dataset(data):
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.shuffle(CONFIG.batch_size)
    ds = ds.batch(CONFIG.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

