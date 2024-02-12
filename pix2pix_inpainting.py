from unittest import result

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
import pathlib
from IPython import display
import datetime
#import cv2
#from sklearn.cluster import KMeans

#region load the dataset
#test 0
resize_x = 256
resize_y = 256


#The bath size of 1 gives better results using the UNet in this experiment.
BATCH_SIZE = 2
EPOCHS = 200
color_channel = 3
crop_size = 150#resize_x / 2


def remove_part_of_image(image):
    radius = np.random.uniform(low=resize_x //5, high=resize_x//3)


    height, width, channels = image.shape
    margin = radius
    center_x = np.random.randint(margin, width - margin)
    center_y = np.random.randint(margin, height - margin)

    #print(f"center_x, center_y = {center_x}, {center_y}")

    y, x = np.ogrid[:height, :width]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 > radius ** 2
    mask = np.repeat(mask[:, :, np.newaxis], channels, axis=2)
    image_with_circle_removed = tf.where(mask, image, tf.zeros_like(image))

    return image_with_circle_removed


#image_type = '*rock_RGB'
image_type = '*oil_drum_RGB'
#image_type = '*clutter_RGB'
#image_type = "*man_made_object_RGB"

train_set_path = pathlib.Path("train")
test_set_path = pathlib.Path("test")

image_paths_train = [str(path) for path in list(train_set_path.glob(image_type + ".jpg"))]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of trainingset: {len(image_paths_train)}")

image_paths_test = [str(path) for path in list(test_set_path.glob(image_type + ".jpg"))]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of testset: {len(image_paths_test)}")

def load_and_preprocess_image(path_image):
    #print("===================start load and preprocess image============================================")
    real_img = tf.io.read_file(path_image)
    real_img = tf.image.decode_jpeg(real_img,
                                 channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
    real_img = tf.cast(real_img, tf.float32)
    real_img = (real_img - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
    real_img = tf.image.resize(real_img, [resize_x, resize_y], method=tf.image.ResizeMethod.AREA)
    input_img = tf.py_function(func = remove_part_of_image, inp = [real_img], Tout=tf.float32)


    return input_img, real_img

#==========================

BUFFER_SIZE = len(image_paths_train)
print(f"BUFFER_SIZE: {BUFFER_SIZE}")

#========================


for image_path in image_paths_train:
    inp,re = load_and_preprocess_image(image_path)
    # plt.figure()
    # plt.title("bilde fra ds")
    # plt.title("inp")
    # plt.imshow(inp)
    # plt.show()

for image_path_test in image_paths_test:
    re_test,inp_test = load_and_preprocess_image(image_path_test)

#======================
# Opprett et tf.data.Dataset fra bildestier
# the dataset consist of both inp and re images.
train_dataset = tf.data.Dataset.from_tensor_slices(image_paths_train)
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


test_dataset = tf.data.Dataset.from_tensor_slices(image_paths_test)
test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

number_of_samples_to_show = BATCH_SIZE  # Antall eksempler du ønsker å vise

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
down_result = down_model(tf.expand_dims(inp,0))
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

LAMBDA = 100
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
    initializer = tf.random_normal_initializer(0., 0.02) #where mean is 0 and the STD is 0.02
    inp = tf.keras.layers.Input(shape=[256,256,3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256,256,3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64,4,False)(x) # fordi vi har en batch size på 128,128,64
    down2 = downsample(128,4)(down1) #batch size 64,64,128
    down3 = downsample(256,4)(down2) #batch size ,32,32,256

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512,4,strides=1, kernel_initializer=initializer,use_bias=True)(zero_pad1) #batch size ,31,31,512
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) #batchsize,33,33,512
    last = tf.keras.layers.Conv2D(1,4,strides=1, kernel_initializer=initializer)(zero_pad2) #batch size 30,30,1
    return tf.keras.Model(inputs=[inp,tar], outputs=[last])

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

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
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

  folder_name = 'generated_images_pix2pix_inpainting'
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

def fit(train_ds, test_ds, steps):
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


fit(train_dataset, test_dataset, steps=40000)

generator.save(f'saved_model_pix2pix_inpainting/{image_type[1:-8]}/my_generator.h5')
discriminator.save(f'saved_model_pix2pix_inpainting/{image_type[1:-8]}/my_discriminator.h5')
#print(f"image_type[1:]: {image_type[1:-8]}")

print('Done!')