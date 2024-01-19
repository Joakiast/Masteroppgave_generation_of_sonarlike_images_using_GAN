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

# Sti til mappen der bildene dine er plassert
train_set_path = "test1"

#list out all files that contain jpg in the path file
image_paths = glob.glob(os.path.join(train_set_path, '*.jpg'))

# Funksjon for å lese og forbehandle bildene
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    # Legg til eventuelle forbehandlingssteg her (resize, normalize, etc.)
    return image

# Last inn bildene som en liste av tensorer
train_images = [load_image(path) for path in image_paths]

# Sjekk dimensjonene til det første bildet
print(train_images[0].shape)

# Vis ett av bildene
plt.figure()
plt.imshow(train_images[0].numpy())
plt.axis('off')  # Skjul koordinataksene
plt.show()

"""
skjelletet til denne koden er ikke ferdig, se DCGAN for ferdig skjelett i github
"""

def make_generator_model():
    """
    sett inn hardkodede tall for parametrene under trening for å unngå for mye beregninger.
    """

    #use_bias = False this is to reduce the models complexity
    #noise parameters
    input_size_noise_x = 7
    input_size_noise_y = 7
    depth_feature_map = 256
    noise_vector = 100

    #convolution parameter
    conv1_filters = 128 # A filter is a matrice of numbers
    conv1_kernel_size = (5,5) # the kernel size is the size of that filter.

    conv2_filters = 64
    conv2_kernel_size = (5, 5)

    conv3_filters = 3
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
    assert model.output_shape == (None, input_size_noise_x*2*2, input_size_noise_y*2*2, conv3_filters) #a test that our image has the expected shape

    return model
def make_discriminator_model():

    size_of_input_image_x = 28
    size_of_input_image_y = 28
    size_of_last_filter_in_generator = 3

#forklaringer til layers
    #med filter så er det antall filteret som brukes over hver epoch og ser de samme pixlene
    #kernel size er størrelsen på dette filteret.
    #strides er hvor mange pixler vi beveger oss bort
    #"Padding='same'" i en Conv2D-lag betyr at outputen beholder samme
    #størrelse som inputen ved å legge til kanter rundt inputen.
    #input shape er størrelsen på det vi sender inn til funksjonen make_discriminator_model
    #altså størrelsen på noise vectoren.

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[size_of_input_image_x, size_of_input_image_y, size_of_last_filter_in_generator]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0],cmap="gray")# cmap='gray'
plt.title("Generated image")
plt.show()
