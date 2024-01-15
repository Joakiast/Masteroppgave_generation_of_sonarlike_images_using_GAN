import tensorflow as tf
tf.__version__


import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display



def make_generator_model():
    #noise parameters
    input_size_noise_x = 7
    input_size_noise_y = 7
    depth_feature_map = 256
    noise_vector = 100

    #convolution parameter
    conv1_filters = 128
    conv1_kernel_size = (5,5)

    conv2_filters = 64
    conv2_kernel_size = (5, 5)

    conv3_filters = 1
    conv3_kernel_size = (5,5)

    model = tf.keras.Sequential()
    model.add(layers.Dense(input_size_noise_x*input_size_noise_y*depth_feature_map, use_bias=False, input_shape=(noise_vector,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((input_size_noise_x, input_size_noise_y, depth_feature_map)))
    assert model.output_shape == (None, input_size_noise_x, input_size_noise_y, depth_feature_map)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(conv1_filters, conv1_kernel_size, strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, input_size_noise_x, input_size_noise_y, conv1_filters)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(conv2_filters, conv2_kernel_size, strides=(2, 2), padding='same', use_bias=False)) #filter reduce stride = 2
    assert model.output_shape == (None, input_size_noise_x*2, input_size_noise_y*2, conv1_filters/2) #strides increase the size (14,14,64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(conv3_filters, conv3_kernel_size, strides=(2, 2), padding='same', use_bias=False, activation='tanh')) # filter = 1 we want black white, rgb conv3 = 3
    assert model.output_shape == (None, input_size_noise_x*2*2, input_size_noise_y*2*2, 1) #a test that our image has the expected shape

    return model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

