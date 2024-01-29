from unittest import result

import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display




log_dir = "logs/"  # Spesifiser ønsket katalog for loggfiler
summary_writer = tf.summary.create_file_writer(log_dir)

start_time = time.time()
# Sti til mappen der bildene dine er plassert
train_set_path = pathlib.Path("train1")
input_set_path = pathlib.Path("test1") # endre denne når simuleringsdata kommer

"""
Dersom jeg ønsker rock så kommenter ut de 2 andre
"""
BATCH_SIZE = 3
#image_type = '*rock_RGB.jpg'
#image_type = '*oil_drum_RGB.jpg'
image_type = '*clutter_RGB.jpg'
EPOCHS = 30

print(f"image_type[1:]: {image_type[1:-8]}")


# Opprett en liste over bildestier som strenger
image_paths = [str(path) for path in list(train_set_path.glob(image_type))]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of trainingset: {len(image_paths)}")
input_image_paths = [str(path) for path in list(input_set_path.glob(image_type))]
print(f"size of input set: {len(input_image_paths)}")


# Funksjon for å lese og forbehandle bildene
resize_x = 28
resize_y = 28

"""
increase the dataset used for "rock and oil"
"""
color_channel = 3
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
    image = tf.image.resize(image, [resize_y, resize_x],method=tf.image.ResizeMethod.AREA) #ønsket resize størrelse, jo mindre jo raskere og dårligere kvalitet
    image = tf.cast(image, tf.float32)
    #print(f"image shape: {image.shape}")
    image = (image - 127.5) /127.5  # Normaliser bildene til [-1, 1] området
    return image

BUFFER_SIZE = len(image_paths)

#print(BUFFER_SIZE)
flipped_images_left_to_right = []  # Opprett en liste for de augmenterte bildene
flipped_images_up_down = []
random_rotated = []
random_cropped_images = []

for image_path in image_paths:
    original_image = load_and_preprocess_image(image_path)
    if "rock_RGB" in image_type or "*oil_drum_RGB.jpg" in image_type:
        #print("Det er rock, gjør noe")
        flip_image_left_right = tf.image.flip_left_right(load_and_preprocess_image(image_path))
        flip_image_up_down = tf.image.flip_up_down(load_and_preprocess_image(image_path))
        random_rotate = tf.image.rot90(load_and_preprocess_image(image_path), k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        random_cropped_image = tf.image.random_crop(load_and_preprocess_image(image_path), size=(tf.shape(load_and_preprocess_image(image_path))[0], tf.shape(load_and_preprocess_image(image_path))[1], tf.shape(load_and_preprocess_image(image_path))[2]))

        flipped_images_left_to_right.append(flip_image_left_right)
        flipped_images_up_down.append(flip_image_up_down)
        random_rotated.append(random_rotate)
        random_cropped_images.append(random_cropped_image)
       # print(f" augmented_images.shape {len(flipped_images_left_to_right)}")
    else:
        pass

# Opprett en tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
input_dataset = tf.data.Dataset.from_tensor_slices(input_set_path)
print(f"dataset shape 1: {len(train_dataset)}")
train_dataset = train_dataset.map(load_and_preprocess_image)
input_dataset = input_dataset.map(load_and_preprocess_image)
print(f"dataset shape 2: {len(train_dataset)}")

if "rock_RGB" in image_type or "*oil_drum_RGB.jpg" in image_type:

    flipped_images_left_right = tf.data.Dataset.from_tensor_slices(flipped_images_left_to_right)
    train_dataset = train_dataset.concatenate(flipped_images_left_right)

    print(f"dataset shape 3: {len(train_dataset)}")
    flipped_images_up_down = tf.data.Dataset.from_tensor_slices(flipped_images_up_down)
    train_dataset = train_dataset.concatenate(flipped_images_up_down)
    print(f"dataset shape 4: {len(train_dataset)}")

    random_rotated = tf.data.Dataset.from_tensor_slices(random_rotated)
    train_dataset = train_dataset.concatenate(random_rotated)
    print(f"dataset shape 5: {len(train_dataset)}")

    random_cropped = tf.data.Dataset.from_tensor_slices(random_cropped_images)
    train_dataset = train_dataset.concatenate(random_cropped)
    print(f"dataset shape 6: {len(train_dataset)}")


train_dataset = train_dataset.shuffle(BUFFER_SIZE)  # Bland datasettet, hvis ønskelig
#input_dataset = input_dataset.shuffle(BUFFER_SIZE)
print(f"dataset shape 7: {len(train_dataset)}")
train_dataset = train_dataset.batch(BATCH_SIZE)
input_dataset = input_dataset.batch(BATCH_SIZE)
print(f"dataset shape 8: {len(train_dataset)}")
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # For ytelsesoptimalisering
input_dataset = input_dataset.prefetch(tf.data.AUTOTUNE)  # For ytelsesoptimalisering

print(f"dataset shape 9: {len(train_dataset)}")

num_batches = len(list(train_dataset))

print("Antall batcher i datasettet:", num_batches)
# Du kan nå iterere over train_dataset i din treningsloop
number_of_samples_show = 2
for images in train_dataset.take(1):  # Ta bare en batch for visning
    plt.figure(figsize=(10, 10))
    for i in range(number_of_samples_show):
        plt.subplot(1, number_of_samples_show, i + 1)
        plt.imshow(images[i])
        plt.axis('on')
        #plt.title(images[i])
        print(images[i].shape)
plt.show()

inp = load_and_preprocess_image(input_image_paths)

#print(f"input shape: {len(inp)}")
# plt.figure(figsize=(10, 10))
# plt.title("input image fra ...")
# plt.imshow(inp)
# plt.show()



#region load the dataset
#===============================fra tensorflow lasting og preprocessing av data===============================
# dataset_name = "facades"
# _URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz"
#
# path_to_zip = tf.keras.utils.get_file(
#     fname = f"{dataset_name}.tar.gz",
#     origin=_URL,
#     extract=True
# )
# path_to_zip = pathlib.Path(path_to_zip)
# PATH = path_to_zip.parent/dataset_name
# list(PATH.parent.iterdir())
#
# sample_image = tf.io.read_file(str(PATH / "train/1.jpg"))
# sample_image = tf.io.decode_jpeg(sample_image)
# print(sample_image.shape)
# plt.figure()
# plt.imshow(sample_image)
# plt.show()
# #endregion
# # since we are getting those two pictures in the one picture we need to divide the pictures in the two parts.
# #region dividing the images
# def load(image_file):
#     image = tf.io.read_file(image_file)
#     image = tf.io.decode_jpeg(image)
#     #splitting the images
#     w = tf.shape(image)[1] # 1 is the 2nd dimention of the picture wich is the width.
#     w = w//2
#     input_image = image[:,w:,:] # input image ==================================================================
#     real_image = image[:,:w,:] #output  image ==================================================================
#     input_image = tf.cast(input_image, tf.float32)
#     real_image = tf.cast(real_image, tf.float32)
#     return input_image, real_image
# #endregion
#
# #region plot the splitted sample
# inp, re = load(str(PATH / "train/100.jpg")) #inp = input image, re = real image
# #casting to int for matplotlib to be able to display the image
# plt.figure()
# plt.imshow(inp/255.0)
# plt.show()
# plt.figure()
# plt.imshow(re / 255.0)
# plt.show()
#
# #endregion
#
# #region prepare for training with training parameters
# #The dataset consist of 400 images in this example
# BUFFER_SIZE = 400
# #The bath size of 1 gives better results using the UNet in this experiment.
# BATCH_SIZE = 1
# #Each img has size 256x256
# IMG_WIDTH = 256
# IMG_HEIGHT = 256
#
# def resize(input_image, real_image, height, width):
#   input_image = tf.image.resize(input_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#   real_image = tf.image.resize(real_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#   return input_image, real_image
# def random_crop(input_image, real_image):
#     stacked_image = tf.stack([input_image, real_image], axis = 0)
#     cropped_image = tf.image.random_crop(stacked_image, size =[2,IMG_HEIGHT,IMG_WIDTH,3])
#     return cropped_image[0],cropped_image[1]
#
# #normalising the imagesd to -1,1
# def normalize(input_image, real_image):
#     input_image = (input_image / 127.5) -1
#     real_image = (real_image / 127.5) -1
#     return input_image, real_image
#
# @tf.function()
# def random_jitter(input_image, real_image):
#   # Resizing to 286x286
#   input_image, real_image = resize(input_image, real_image, 286, 286)
#
#   # Random cropping back to 256x256
#   input_image, real_image = random_crop(input_image, real_image)
#
#   if tf.random.uniform(()) > 0.5:
#     # Random mirroring
#     input_image = tf.image.flip_left_right(input_image)
#     real_image = tf.image.flip_left_right(real_image)
#
#   return input_image, real_image
#
# plt.figure(figsize=(6, 6))
# for i in range(4):
#   rj_inp, rj_re = random_jitter(inp, re)
#   plt.subplot(2, 2, i + 1)
#   plt.imshow(rj_inp / 255.0)
#   plt.axis('off')
# plt.show()
#
# def load_image_train(image_file):
#     input_image, real_image = load(image_file)
#     input_image, real_image = random_jitter(input_image, real_image)
#     input_image, real_image = normalize(input_image, real_image)
#     return input_image, real_image
#
# def load_imge_test(image_file):
#     input_image, real_image = load(image_file)
#     input_image, real_image = resize(input_image, real_image,IMG_HEIGHT,IMG_WIDTH)
#     input_image, real_image = normalize(input_image, real_image)
#     return input_image, real_image
#
# #endregion
#
# #region build input pipeline with tf.data
# train_dataset = tf.data.Dataset.list_files(str(PATH / "train/*.jpg"))
# train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
# train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# train_dataset = train_dataset.batch(BATCH_SIZE)
#
# try:
#     test_dataset = tf.data.Dataset.list_files(str(PATH / "test/*.jpg"))
# except tf.errors.InvalidArgumentError:
#     test_dataset = tf.data.Dataset.list_files(str(PATH / "val/*.jpg"))
# test_dataset = test_dataset.map(load_imge_test)
# test_dataset=test_dataset.batch(BATCH_SIZE)
# #endregion
#
#===============================fra tensorflow lasting og preprocessing av data===============================

# #region Createing the generator

"""
The generator of your pix2pix cGAN is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). (You can find out more about it in the Image segmentation tutorial and on the U-Net project website.)

Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU
Each block in the decoder is: Transposed convolution -> Batch normalization -> Dropout (applied to the first 3 blocks) -> ReLU
There are skip connections between the encoder and decoder (as in the U-Net).
Define the downsampler (encoder):
"""
# OUTPUT_CHANNELS = 3
#
# def downsample(filters,size,apply_batchnorm = True):
#     initializer = tf.random_normal_initializer(0., 0.02)
#     result = tf.keras.Sequential()
#     result.add(tf.keras.layers.Conv2D(filters,size,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
#     if apply_batchnorm:
#         result.add(tf.keras.layers.BatchNormalization())
#     result.add(tf.keras.layers.LeakyReLU())
#     return result
#
# down_model = downsample(3,4)
# down_result = down_model(tf.expand_dims(inp,0))
# print(down_result.shape)
#
# def upsample(filters,size,apply_dropout = False):
#     initializer = tf.random_normal_initializer(0., 0.02)
#     result = tf.keras.Sequential()
#     result.add(tf.keras.layers.Conv2DTranspose(filters,size,strides=2,padding='same'
#                                                ,kernel_initializer=initializer,use_bias=False))
#     result.add(tf.keras.layers.BatchNormalization())
#     if apply_dropout:
#         result.add(tf.keras.layers.Dropout(0.5))
#     result.add(tf.keras.layers.ReLU())
#     return result
#
# up_model = upsample(3,4)
# up_result = up_model(down_result)
# print(up_result.shape)
#
#
# def Generator():
#   inputs = tf.keras.layers.Input(shape=[256, 256, 3])
#
#   down_stack = [
#     downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
#     downsample(128, 4),  # (batch_size, 64, 64, 128)
#     downsample(256, 4),  # (batch_size, 32, 32, 256)
#     downsample(512, 4),  # (batch_size, 16, 16, 512)
#     downsample(512, 4),  # (batch_size, 8, 8, 512)
#     downsample(512, 4),  # (batch_size, 4, 4, 512)
#     downsample(512, 4),  # (batch_size, 2, 2, 512)
#     downsample(512, 4),  # (batch_size, 1, 1, 512)
#   ]
#
#   up_stack = [
#     upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
#     upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
#     upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
#     upsample(512, 4),  # (batch_size, 16, 16, 1024)
#     upsample(256, 4),  # (batch_size, 32, 32, 512)
#     upsample(128, 4),  # (batch_size, 64, 64, 256)
#     upsample(64, 4),  # (batch_size, 128, 128, 128)
#   ]
#
#   initializer = tf.random_normal_initializer(0., 0.02)
#   last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
#                                          strides=2,
#                                          padding='same',
#                                          kernel_initializer=initializer,
#                                          activation='tanh')  # (batch_size, 256, 256, 3)
#
#   x = inputs
#
#   # Downsampling through the model
#   skips = []
#   for down in down_stack:
#     x = down(x)
#     skips.append(x)
#
#   skips = reversed(skips[:-1])
#
#   # Upsampling and establishing the skip connections
#   for up, skip in zip(up_stack, skips):
#     x = up(x)
#     x = tf.keras.layers.Concatenate()([x, skip])
#
#   x = last(x)
#
#   return tf.keras.Model(inputs=inputs, outputs=x)
#
# generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
#
# gen_output = generator(inp[tf.newaxis, ...], training=False)
# plt.imshow(gen_output[0, ...])
# plt.show()
#
# LAMBDA = 100
# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#
# def generator_loss(disc_genrerated_output,get_output, target):
#     gan_loss = loss_object(tf.ones_like(disc_genrerated_output),disc_genrerated_output)
#
#     #mean absolute error
#
#     l1_loss = tf.reduce_mean(tf.abs(target - get_output))
#     total_gen_loss = gan_loss + (LAMBDA * l1_loss)
#     return total_gen_loss,gan_loss,l1_loss
#
# #endregion
#
# #regi0on build the discriminator
#
# def Discriminator():
#     initializer = tf.random_normal_initializer(0., 0.02) #where mean is 0 and the STD is 0.02
#     inp = tf.keras.layers.Input(shape=[256,256,3], name='input_image')
#     tar = tf.keras.layers.Input(shape=[256,256,3], name='target_image')
#     x = tf.keras.layers.concatenate([inp, tar])
#     down1 = downsample(64,4,False)(x) # fordi vi har en batch size på 128,128,64
#     down2 = downsample(128,4)(down1) #batch size 64,64,128
#     down3 = downsample(256,4)(down2) #batch size ,32,32,256
#
#     zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
#     conv = tf.keras.layers.Conv2D(512,4,strides=1, kernel_initializer=initializer,use_bias=False)(zero_pad1) #batch size ,31,31,512
#     batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
#     leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
#     zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) #batchsize,33,33,512
#     last = tf.keras.layers.Conv2D(1,4,strides=1, kernel_initializer=initializer)(zero_pad2) #batch size 30,30,1
#     return tf.keras.Model(inputs=[inp,tar], outputs=[last])
#
# discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True,dpi=64)
#
# disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
# plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
# plt.colorbar()
# plt.show()
# # #endregion
#
# #region define discriminator loss
# def discriminator_loss(disc_real_output,disc_generated_output):
#     real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
#     generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
#     total_disc_loss = real_loss + generated_loss
#     return total_disc_loss
#
# #endregion
#
# #region Optimizer and checkpoint saver
#
# generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
#
# # ckeckpoint_dir = "./training_checkpoints"
# # checkpoint_prefix = os.path.join(ckeckpoint_dir, "ckpt")
# # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,
# #                                  generator=generator, discriminator=discriminator)
#
# #endregion
#
# #region generate images
#
# def generate_images(model, test_input, tar):
#   prediction = model(test_input, training=True)
#   plt.figure(figsize=(15, 15))
#
#   display_list = [test_input[0], tar[0], prediction[0]]
#   title = ['Input Image', 'Ground Truth', 'Predicted Image']
#
#   for i in range(3):
#     plt.subplot(1, 3, i+1)
#     plt.title(title[i])
#     # Getting the pixel values in the [0, 1] range to plot.
#     plt.imshow(display_list[i] * 0.5 + 0.5)
#     plt.axis('off')
#   plt.show()
#
# for example_input, example_target in test_dataset.take(1):
#   generate_images(generator, example_input, example_target)
# #endregion
#
#
# log_dir="logs/"
#
# summary_writer = tf.summary.create_file_writer(
#   log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#
#
# @tf.function
# def train_step(input_image, target, step):
#   with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#     gen_output = generator(input_image, training=True)
#
#     disc_real_output = discriminator([input_image, target], training=True)
#     disc_generated_output = discriminator([input_image, gen_output], training=True)
#
#     gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
#     disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
#
#   generator_gradients = gen_tape.gradient(gen_total_loss,
#                                           generator.trainable_variables)
#   discriminator_gradients = disc_tape.gradient(disc_loss,
#                                                discriminator.trainable_variables)
#
#   generator_optimizer.apply_gradients(zip(generator_gradients,
#                                           generator.trainable_variables))
#   discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
#                                               discriminator.trainable_variables))
#
#   with summary_writer.as_default():
#     tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
#     tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
#     tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
#     tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
#
# def fit(train_ds, test_ds, steps):
#   example_input, example_target = next(iter(test_ds.take(1)))
#   start = time.time()
#
#   for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
#     if (step) % 1000 == 0:
#       display.clear_output(wait=True)
#
#       if step != 0:
#         print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
#
#       start = time.time()
#
#       generate_images(generator, example_input, example_target)
#       print(f"Step: {step//1000}k")
#
#     train_step(input_image, target, step)
#
#     # Training step
#     if (step+1) % 10 == 0:
#       print('.', end='', flush=True)
#
#
#     # # Save (checkpoint) the model every 5k steps
#     # if (step + 1) % 5000 == 0:
#     #   checkpoint.save(file_prefix=checkpoint_prefix)
#
# def fit(train_ds, test_ds, steps):
#   example_input, example_target = next(iter(test_ds.take(1)))
#   start = time.time()
#
#   for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
#     if (step) % 1000 == 0:
#       display.clear_output(wait=True)
#
#       if step != 0:
#         print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')
#
#       start = time.time()
#
#       generate_images(generator, example_input, example_target)
#       print(f"Step: {step//1000}k")
#
#     train_step(input_image, target, step)
#
#     # Training step
#     if (step+1) % 10 == 0:
#       print('.', end='', flush=True)
#
#
#     # # Save (checkpoint) the model every 5k steps
#     # if (step + 1) % 5000 == 0:
#     #   checkpoint.save(file_prefix=checkpoint_prefix)
#
# #%load_ext tensorboard
# #%tensorboard --logdir {log_dir}
#
# fit(train_dataset, test_dataset, steps=40000)