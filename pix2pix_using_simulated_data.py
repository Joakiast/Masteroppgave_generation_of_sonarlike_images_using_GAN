from unittest import result

import tensorflow as tf
tf.__version__
import glob
#import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import pathlib
from IPython import display
import datetime
import sys
#import cv2
#from sklearn.cluster import KMeans

print(datetime.datetime.now())
start_time = time.time()
#region load the dataset
#test 0
resize_x = 256
resize_y = 256

#The bath size of 1 gives better results using the UNet in this experiment.
BATCH_SIZE = 1
EPOCHS = 200
color_channel = 3
crop_size = 150#resize_x / 2

# def find_coordinates_for_circle_tensor(path_image):
#     #base_name = str[path_image]#tf.cast(path_image, str)
#     base_name_b = os.path.basename(path_image.numpy())
#     base_name = base_name_b.decode("utf-8")
#     #print(f"base name {base_name}")
#     label_file = base_name.replace('.jpg', '.txt')  # Bytt ut filendelsen fra .jpg til .txt
#     #print(f"label file {label_file}")
#
#     label_path = os.path.join("datasets/train/Label", label_file)
#     #print(f"label_path {label_path}")
#     x, y = None, None
#     try:
#
#         with open(label_path, 'r') as file:
#             label_content = file.read()
#
#         for line in label_content.split('\n'):
#             parts = line.split()
#             if parts and parts[0] != 'clutter':
#                 x, y = map(float, parts[1:3])
#                 #print(f"x: {x}, y: {y}")
#                 return x, y
#
#     except Exception as e:
#         print(f"Error while processing label file {label_path}: {e}")
#     return None, None
#
# def find_coordinates_for_circle(path_image):
#
#     base_name = os.path.basename(path_image)#.numpy())
#     #print(f"base name {base_name}")
#     label_file = base_name.replace('.jpg', '.txt')  # Bytt ut filendelsen fra .jpg til .txt
#     #print(f"label file {label_file}")
#
#     label_path = os.path.join("datasets/train/Label", label_file)
#     #print(f"label_path {label_path}")
#     x, y = None, None
#     try:
#
#         with open(label_path, 'r') as file:
#             label_content = file.read()
#
#         for line in label_content.split('\n'):
#             parts = line.split()
#             if parts and parts[0] != 'clutter':
#                 x, y = map(float, parts[1:3])
#                 #print(f"x: {x}, y: {y}")
#                 return x, y
#
#
#     except Exception as e:
#         print(f"Error while processing label file {label_path}: {e}")
#     return None, None
#
#
#
# def remove_part_of_image(image, point_x, point_y):
#     radius = 50# np.random.uniform(low=resize_x //5, high=resize_x//3)
#
#     height, width, channels = image.shape
#     margin = radius
#     center_x = point_x #np.random.randint(margin, width - margin)
#     center_y = point_y #np.random.randint(margin, height - margin)
#
#     #print(f"center_x, center_y = {center_x}, {center_y}")
#
#     y, x = np.ogrid[:height, :width]
#     mask = (x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2
#     mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)
#     image_with_circle_removed = tf.where(mask, image, tf.zeros_like(image))
#
#     return image_with_circle_removed


#image_type = '*rock_RGB'
image_type = '*oil_drum_RGB'
#image_type = '*clutter_RGB'
#image_type = "*man_made_object_RGB"

train_set_path = pathlib.Path("datasets/train")
train_set_path_simulated = pathlib.Path("datasets/sim_data_rgb_barrel")
test_set_path = pathlib.Path("datasets/test")

image_paths_train = [str(path) for path in list(train_set_path.glob(image_type + ".jpg"))]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of trainingset: {len(image_paths_train)}")

image_paths_train_simulated = [str(path) for path in list(train_set_path_simulated.glob("*.png"))][:len(image_paths_train)]   # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of trainingset: {len(image_paths_train_simulated)}")

image_paths_test = [str(path) for path in list(test_set_path.glob(image_type + ".jpg"))]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of testset: {len(image_paths_test)}")

# def load_and_preprocess_image(path_image):
#     #print("===================start load and preprocess image============================================")
#     real_img = tf.io.read_file(path_image)
#     real_img = tf.image.decode_jpeg(real_img,
#                                  channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
#     real_img = tf.cast(real_img, tf.float32)
#     real_img = (real_img - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
#     x,y = tf.py_function(func = find_coordinates_for_circle, inp = path_image, Tout = (tf.float32, tf.float32))
#
#     real_img = tf.image.resize(real_img, [resize_x, resize_y], method=tf.image.ResizeMethod.AREA)
#     input_img = tf.py_function(func = remove_part_of_image, inp = [real_img,x,y], Tout=tf.float32)
#
#     return input_img, real_img


def crop_image_around_POI(image, point_x, point_y, crop_size):


    # Konverter punktkoordinater til heltall
    point_x = tf.cast(point_x, tf.int32)
    point_y = tf.cast(point_y, tf.int32)
    crop_size = tf.cast(crop_size, tf.int32)

    # Beregn øvre venstre hjørne av beskjæringsboksen
    start_y = tf.maximum(0, point_y - crop_size // 2)
    start_x = tf.maximum(0, point_x - crop_size // 2)

    # Sørg for at beskjæringsboksen ikke går utenfor bildet
    image_height, image_width, _ = image.shape
    image_height = tf.cast(image_height, tf.int32)
    image_width = tf.cast(image_width, tf.int32)

    if start_x + crop_size > image_width:
        start_x = tf.maximum(0, image_width - crop_size)
    if start_y + crop_size > image_height:
        start_y = tf.maximum(0, image_height - crop_size)

    image = tf.image.crop_to_bounding_box(image, start_y, start_x, crop_size, crop_size)

    #print(f"crop by {crop_size}")
    return image

def find_coordinates_for_cropping_tensor(path_image):
    #base_name = str[path_image]#tf.cast(path_image, str)
    base_name_b = os.path.basename(path_image.numpy())
    base_name = base_name_b.decode("utf-8")
    #print(f"base name {base_name}")
    label_file = base_name.replace('.jpg', '.txt')  # Bytt ut filendelsen fra .jpg til .txt
    #print(f"label file {label_file}")

    label_path = os.path.join("datasets/train/Label", label_file)
    #print(f"label_path {label_path}")
    x, y = None, None
    try:

        with open(label_path, 'r') as file:
            label_content = file.read()

        for line in label_content.split('\n'):
            parts = line.split()
            if parts and parts[0] != 'clutter':
                x, y = map(float, parts[1:3])
                #print(f"x: {x}, y: {y}")
                return x, y

    except Exception as e:
        print(f"Error while processing label file {label_path}: {e}")
    return None, None

def find_coordinates_for_cropping(path_image):

    base_name = os.path.basename(path_image)#.numpy())
    #print(f"base name {base_name}")
    label_file = base_name.replace('.jpg', '.txt')  # Bytt ut filendelsen fra .jpg til .txt
    #print(f"label file {label_file}")

    label_path = os.path.join("datasets/train/Label", label_file)
    #print(f"label_path {label_path}")
    x, y = None, None
    try:

        with open(label_path, 'r') as file:
            label_content = file.read()

        for line in label_content.split('\n'):
            parts = line.split()
            if parts and parts[0] != 'clutter':
                x, y = map(float, parts[1:3])
                #print(f"x: {x}, y: {y}")
                return x, y
            # elif parts and parts[0] == 'rock':
            #     x, y = map(float, parts[1:3])
            #     return x,y

    except Exception as e:
        print(f"Error while processing label file {label_path}: {e}")
    return None, None


#endregion


#==========================
def load_and_preprocess_image(path_image_trainset, path_simulated_image_trainset):

    #path_simulated_image_trainset = pathlib.Path("datasets")
    if isinstance(path_image_trainset, tf.Tensor):
        #print("===================start load and preprocess image============================================")
        image = tf.io.read_file(path_image_trainset)
        image = tf.image.decode_jpeg(image,
                                     channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde

        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
        if not "clutter" in image_type:
            #x,y = tf.py_function(func=find_coordinates_for_circle_tensor, inp=[path_image_trainset], Tout=[tf.float32, tf.float32])
            x, y = tf.py_function(func=find_coordinates_for_cropping_tensor, inp=[path_image_trainset],
                                  Tout=[tf.float32, tf.float32])
            image.set_shape([400, 600, 3])
            image = crop_image_around_POI(image, x, y, crop_size)
            #inp_image = tf.py_function(func = remove_part_of_image, inp = [image,x,y], Tout = tf.float32)#remove_part_of_image(image,x,y)#crop_image_around_POI(image, x, y, crop_size)
            #inp_image.set_shape([400, 600, 3])
        image = tf.image.resize(image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
#====================for inp===============================================================================
        inp_image = tf.io.read_file(path_simulated_image_trainset)
        inp_image = tf.image.decode_png(inp_image,
                                     channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde

        inp_image = tf.cast(inp_image, tf.float32)
        inp_image = (inp_image - 127.5)/127.5
        if not "clutter" in image_type:
            inp_image.set_shape([369,496,3])
        inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)




        #inp_image = tf.image.resize(inp_image , [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
        return inp_image, image
    else:
        #print("===================start load and preprocess image============================================")
        image = tf.io.read_file(path_image_trainset)
        image = tf.image.decode_jpeg(image,
                                     channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
        assert image.shape == (400, 600, 3)
        #image = tf.image.resize(image, [400, 600], method=tf.image.ResizeMethod.AREA)
        if not "clutter" in image_type:
            x, y = find_coordinates_for_cropping(path_image_trainset)
            image = crop_image_around_POI(image, x, y, crop_size)
            # image = tf.image.resize(image, [resize_x,resize_y], method=tf.image.ResizeMethod.AREA)
            # print(f"alle bilder kommer hit: image shape før resize: {image.shape} bilde: {path_image}")
        image = tf.image.resize(image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)

        #=================================for inp image======================================================
        inp_image = tf.io.read_file(path_simulated_image_trainset)
        inp_image = tf.image.decode_png(inp_image,channels=color_channel)
        #print(f" inp_image.shape: {inp_image.shape}")
        inp_image = tf.cast(inp_image, tf.float32)
        inp_image = (inp_image - 127.5) / 127.5
        assert inp_image.shape == (369,496,3)
        inp_image = tf.image.resize(inp_image, [resize_x,resize_y], method=tf.image.ResizeMethod.LANCZOS5)




        return inp_image, image
        # else:
        #     return image

BUFFER_SIZE = len(image_paths_train)
print(f"BUFFER_SIZE: {BUFFER_SIZE}")

#========================
#============================================================================


def augmentation(input_img, real_img):

    flipped_left_right = (tf.image.flip_left_right(input_img), tf.image.flip_left_right(real_img))
    flipped_up_down = (tf.image.flip_up_down(input_img), tf.image.flip_up_down(real_img))
    rotate = (tf.image.rot90(input_img), tf.image.rot90(real_img))

    return flipped_left_right, flipped_up_down, rotate


inp_augmented_training_data_flip_left_right = []
real_augmented_training_data_flip_left_right = []

inp_augmented_training_data_flip_up_down = []
real_augmented_training_data_flip_up_down = []

inp_augmented_training_data_rotate = []
real_augmented_training_data_rotate = []

i = 0
# for image_path in image_paths_train:# and image_paths_simulated in image_paths_train_simulated:
#     inp, re = load_and_preprocess_image(image_path,image_paths_simulated)
for image_path, image_path_simulated in zip(image_paths_train, image_paths_train_simulated):
    inp, re = load_and_preprocess_image(image_path, image_path_simulated)
    #
    #
    # # Konverterer begge tensorer til uint8 ved å bruke tf.cast etter å ha skalert dem
    # inp_uint8 = tf.cast(inp * 255, tf.uint8)
    # re_uint8 = tf.cast(re * 255, tf.uint8)
    #
    # # Konverterer tensorer til numpy-arrays for å kunne vises med matplotlib
    # inp_numpy = inp_uint8.numpy()
    # re_numpy = re_uint8.numpy()
    #
    # plt.figure(figsize=(10, 5))
    #
    # plt.subplot(1, 2, 1)
    # plt.title("inp")
    # plt.imshow(inp_numpy)
    #
    # plt.subplot(1, 2, 2)
    # plt.title("re")
    # plt.imshow(re_numpy)
    #
    # plt.show()
    #
    # i += 1  # Øker telleren for hver iterasjon
    # if i > 9:
    #     sys.exit()
    # #tf.py_function(func=find_coordinates_for_cropping_tensor, inp=[path_image], Tout=[tf.float32,tf.float32])
    # plt.show()
    if "rock_RGB" in image_type or "oil_drum_RGB" in image_type or "man_made_object_RGB" in image_type:
        flipped_left_right, flipped_up_down, rotate = augmentation(inp,re)#tf.py_function(func = augmentation, inp = [inp, re], Tout=[tf.float32,tf.float32,tf.float32])

        flipped_left_right_inp, flipped_left_right_real = flipped_left_right
        inp_augmented_training_data_flip_left_right.append(flipped_left_right_inp)  # , flipped_up_down, rotate])
        real_augmented_training_data_flip_left_right.append(flipped_left_right_real)  # , flipped_up_down, rotate])

        flipped_up_down_inp, flipped_up_down_real = flipped_up_down
        inp_augmented_training_data_flip_up_down.append(flipped_up_down_inp)
        real_augmented_training_data_flip_up_down.append(flipped_up_down_real)

        rotate_inp, rotate_real = rotate
        inp_augmented_training_data_rotate.append(rotate_inp)
        real_augmented_training_data_rotate.append(rotate_real)

    elif "clutter" in image_type:
        pass

# for image_path_test in image_paths_test:
#     re_test,inp_test = load_and_preprocess_image(image_path_test)


#======================
# Opprett et tf.data.Dataset fra bildestier
# the dataset consist of both inp and re images.
train_dataset = tf.data.Dataset.from_tensor_slices((image_paths_train,image_paths_train_simulated))
print(image_paths_train[0])
print(f"dataset shape 1: {len(train_dataset)}")

train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
print(train_dataset)

augmented_training_dataset_flip_left_right = tf.data.Dataset.from_tensor_slices((inp_augmented_training_data_flip_left_right, real_augmented_training_data_flip_left_right))
train_dataset = train_dataset.concatenate(augmented_training_dataset_flip_left_right)
print(f"dataset shape 2: {len(train_dataset)}")

augmented_training_dataset_flip_up_down = tf.data.Dataset.from_tensor_slices((inp_augmented_training_data_flip_up_down, real_augmented_training_data_flip_up_down))
train_dataset = train_dataset.concatenate(augmented_training_dataset_flip_up_down)
print(f"dataset shape 3: {len(train_dataset)}")

augmented_training_dataset_rotate = tf.data.Dataset.from_tensor_slices((inp_augmented_training_data_rotate, real_augmented_training_data_rotate))
train_dataset = train_dataset.concatenate(augmented_training_dataset_rotate)
print(f"dataset shape 4: {len(train_dataset)}")

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
print(f"dataset shape 5: {len(train_dataset)}")
train_dataset = train_dataset.batch(BATCH_SIZE)
print(f"dataset shape 6: {len(train_dataset)}")
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


# test_dataset = tf.data.Dataset.from_tensor_slices(image_paths_test)
# test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
# test_dataset = test_dataset.shuffle(BUFFER_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)
# test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

#============================================================================




number_of_samples_to_show = BATCH_SIZE  # Antall eksempler du ønsker å vise


for i in train_dataset.take(1):
    print(f"Element type tuple len: {len(i)}")

# Tar en batch fra datasettet
for input_imgs, real_imgs in train_dataset.take(1):
    plt.figure(figsize=(10, 5))
    for i in range(number_of_samples_to_show):
        # Plotter input_img
        ax = plt.subplot(2, number_of_samples_to_show, 2*i + 1)
        plt.title("Input Image")
        plt.imshow(input_imgs[i].numpy() )

        # Plotter real_img
        ax = plt.subplot(2, number_of_samples_to_show, 2*i + 2)
        plt.title("Real Image")
        plt.imshow(real_imgs[i].numpy())
        plt.axis('on')

plt.tight_layout()
plt.show()

#======================
#endregion

#region Createing the generator

"""
The generator of your pix2pix cGAN is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). (You can find out more about it in the Image segmentation tutorial and on the U-Net project website.)

Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU
Each block in the decoder is: Transposed convolution -> Batch normalization -> Dropout (applied to the first 3 blocks) -> ReLU
There are skip connections between the encoder and decoder (as in the U-Net).
Define the downsampler (encoder):
"""
OUTPUT_CHANNELS = 3

def downsample(filters,size,apply_batchnorm = True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=True))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

down_model = downsample(3,4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)

def upsample(filters,size,apply_dropout = False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters,size,strides=2,padding='same'
                                               ,kernel_initializer=initializer,use_bias=True))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

up_model = upsample(3,4)
up_result = up_model(down_result)
print(up_result.shape)


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])
plt.title("testing the generated output ")
plt.show()

LAMBDA = 100 # jo større lambda er jo høyere skal likheten mellom treningsdataene og det genererte bilde være.
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_genrerated_output,get_output, target):
    gan_loss = loss_object(tf.ones_like(disc_genrerated_output),disc_genrerated_output)

    #mean absolute error

    l1_loss = tf.reduce_mean(tf.abs(target - get_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss,gan_loss,l1_loss

#endregion

#regi0on build the discriminator
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True,dpi=64)

disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
plt.show()
# #endregion

#region define discriminator loss
def discriminator_loss(disc_real_output,disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

#endregion

#region Optimizer and checkpoint saver

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.6)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.6)
#
# ckeckpoint_dir = "./training_checkpoints"
# checkpoint_prefix = os.path.join(ckeckpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator, discriminator=discriminator)
#
# #endregion
#
# #region generate images

def generate_images(model, test_input, tar,step):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))
  diff = prediction - tar
  display_list = [test_input[0], tar[0], prediction[0], diff[0]]
  title = ['Input Image', 'Real Image', 'Generated Image', 'Difference']

  num_elem = len(display_list)

  for i in range(num_elem):
    plt.subplot(1, num_elem, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')

  folder_name = 'generated_data/generated_images_pix2pix_simulated_dataset'
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)

  # Save the figure using the step number to keep track of progress
  plt.savefig(f'{folder_name}/image_at_step_{step//1000:04d}.png')
  #plt.close()  # Close the figure to free up memory
  #print('Saved generated images at step '+ str(step))
  plt.show()




# for example_input, example_target in test_dataset.take(1):
#   generate_images(generator, example_input, example_target,step=0)
# #endregion


log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, steps):
  # example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()
      #
      generate_images(generator, input_image, target,step)
      print(f"Step: {step//1000}k")
      #
      # folder_name = 'generated_images'
      # if not os.path.exists(folder_name):
      #     os.makedirs(folder_name)
      #
      #
      # plt.savefig(os.path.join(folder_name, ' image_at_step_{:04d}.png'.format(step//1000)))
      # print('fig closed')
      # plt.close("all")
      # # plt.show() plot for hver epoch
      # # plt.savefig(‘din_fig.png’)

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # # Save (checkpoint) the model every 5k steps
    # if (step + 1) % 5000 == 0:
    #   checkpoint.save(file_prefix=checkpoint_prefix)


fit(train_dataset, steps=100000)

generator.save(f'saved_model_pix2pix_inpainting/{image_type[1:-8]}/my_generator.h5')
discriminator.save(f'saved_model_pix2pix_inpainting/{image_type[1:-8]}/my_discriminator.h5')
#print(f"image_type[1:]: {image_type[1:-8]}")

end_time = time.time()  # Lagrer slutttiden
elapsed_time = end_time - start_time  # Beregner tiden det tok å kjøre koden

print(f"Tiden det tok å kjøre koden: {elapsed_time/60} minutter")
print('Done!')