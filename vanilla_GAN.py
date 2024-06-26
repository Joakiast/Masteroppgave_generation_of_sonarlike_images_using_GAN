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
import neptune
from neptune.types import File
run = neptune.init_run(
    project="masteroppgave/vanilla-GAN",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMDY3ZDFlNS1hMGVhLTQ1N2YtODg4MC1hNThiOTM1NGM3YTQifQ=="
)



log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir)

start_time = time.time()


train_set_path = pathlib.Path("datasets/train")
train_set_label_path = pathlib.Path("datasets/train/Label")

"""
Dersom jeg ønsker rock, så kommenter ut de 2 andre
"""
BATCH_SIZE = 3
#image_type = '*rock_RGB'
image_type = '*oil_drum_RGB'
#image_type = '*clutter_RGB'
#image_type = "*man_made_object_RGB"
EPOCHS = 2000

print(f"image_type[1:]: {image_type[1:-4]}" )


# Opprett en liste over bildestier som strenger
image_paths = [str(path) for path in list(train_set_path.glob(image_type +".jpg"))]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of trainingset: {len(image_paths)}")
label_path = [str(path) for path in list(train_set_label_path.glob(image_type +".txt"))]
# Funksjon for å lese og forbehandle bildene
resize_x = 256
resize_y = 256
crop_size = 150#resize_x / 2
noise_vector_for_gen = 512 #husk å endre inni generator

if image_type != 'clutter':
    print(f"image_type: {image_type} crop size: {crop_size}")

"""
increase the dataset used for "rock and oil"
"""
color_channel = 3

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
def load_and_preprocess_image(path_image):
    if isinstance(path_image, tf.Tensor):
        #print("===================start load and preprocess image============================================")
        image = tf.io.read_file(path_image)
        image = tf.image.decode_jpeg(image,
                                     channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde

        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
        if not "clutter" in image_type:
            x,y = tf.py_function(func=find_coordinates_for_cropping_tensor, inp=[path_image], Tout=[tf.float32,tf.float32])
            image.set_shape([400, 600, 3])
            image = crop_image_around_POI(image, x, y, crop_size)
        image = tf.image.resize(image, [resize_x, resize_y], method=tf.image.ResizeMethod.AREA)
        return image
    else:
        #print("===================start load and preprocess image============================================")
        image = tf.io.read_file(path_image)
        image = tf.image.decode_jpeg(image,
                                     channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
        assert image.shape == (400, 600, 3)
        #image = tf.image.resize(image, [400, 600], method=tf.image.ResizeMethod.AREA)

        if not "clutter" in image_type:
            x,y = find_coordinates_for_cropping(path_image)
            image = crop_image_around_POI(image, x, y, crop_size)
        #image = tf.image.resize(image, [resize_x,resize_y], method=tf.image.ResizeMethod.AREA)
        #print(f"alle bilder kommer hit: image shape før resize: {image.shape} bilde: {path_image}")
        image = tf.image.resize(image, [resize_x, resize_y], method=tf.image.ResizeMethod.AREA)
        return image
        # else:
        #     return image


BUFFER_SIZE = len(image_paths)

#print(BUFFER_SIZE)
flipped_images_left_to_right = []
flipped_images_up_down = []
random_rotated = []
#random_cropped_images = []

for image_path in image_paths:
    original_image = load_and_preprocess_image(image_path)
#     plt.figure()
#     plt.title("Croppet bilde")
#     plt.imshow(original_image)
#     plt.show()

    if "rock_RGB" in image_type or "oil_drum_RGB" in image_type or "man_made_object_RGB" in image_type:
        # Påfør transformasjoner direkte på `original_image`
        flip_image_left_right = tf.image.flip_left_right(original_image)
        flip_image_up_down = tf.image.flip_up_down(original_image)
        random_rotate = tf.image.rot90(original_image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

        # Legg til de transformerte bildene i de respektive listene
        flipped_images_left_to_right.append(flip_image_left_right)
        flipped_images_up_down.append(flip_image_up_down)
        random_rotated.append(random_rotate)
    elif "clutter" in image_type:
        pass

# Opprett en tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
print(f"dataset shape 1: {len(train_dataset)}")
train_dataset = train_dataset.map(load_and_preprocess_image)
print(f"dataset shape 2: {len(train_dataset)}")

if "rock_RGB" in image_type or "*oil_drum_RGB" in image_type or "man_made_object_RGB" in image_type:

    flipped_images_left_right = tf.data.Dataset.from_tensor_slices(flipped_images_left_to_right)
    train_dataset = train_dataset.concatenate(flipped_images_left_right)

    print(f"dataset shape 3: {len(train_dataset)}")
    flipped_images_up_down = tf.data.Dataset.from_tensor_slices(flipped_images_up_down)
    train_dataset = train_dataset.concatenate(flipped_images_up_down)
    print(f"dataset shape 4: {len(train_dataset)}")

    random_rotated = tf.data.Dataset.from_tensor_slices(random_rotated)
    train_dataset = train_dataset.concatenate(random_rotated)
    print(f"dataset shape 5: {len(train_dataset)}")

    #random_cropped = tf.data.Dataset.from_tensor_slices(random_cropped_images)
    #train_dataset = train_dataset.concatenate(random_cropped)
    print(f"dataset shape 6: {len(train_dataset)}")
else:
    pass


train_dataset = train_dataset.shuffle(BUFFER_SIZE)  # Bland datasettet, hvis ønskelig, med størrelesn lik antal bilder
print(f"dataset shape 7: {len(train_dataset)}")
train_dataset = train_dataset.batch(BATCH_SIZE)
print(f"dataset shape 8: {len(train_dataset)}")
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # For ytelsesoptimalisering
print(f"dataset shape 9: {len(train_dataset)}")
num_batches = len(list(train_dataset))

print("Antall batcher i datasettet:", num_batches)
# Du kan nå iterere over train_dataset i din treningsloop
number_of_samples_show = 3
for images in train_dataset.take(1):  # Ta bare en batch for visning
    plt.figure(figsize=(10, 10))
    for i in range(number_of_samples_show):
        plt.subplot(1, number_of_samples_show, i + 1)
        plt.title("plot of images in dataset")
        plt.imshow(images[i])
        plt.axis('on')
        #plt.title(images[i])
        print(images[i].shape)
plt.show()
###kommentert ut alt over
#lagt inn kode mellom ====

#====================================
#=======================================med mnist dataset husk å endre size_of_last_filter_in_generator til 1 og conv3_filters til 1
# start_time = time.time()
# print("======================Start======================")
#
# (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
#
# """
# print(train_images.shape)
# plt.plot(train_images[60000-1,:,:])
# plt.show()
# """
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
#
# print(train_images.shape)
# plt.imshow(train_images[60000-1,:,:,0])  # Plotting the last image
# #plt.show()
# #plt.savefig(‘din_fig1.png’)
# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
#
# plt.imshow(train_images[60000-1,:,:,0])  # Plotting the last image
# #plt.show()
# #plt.savefig(‘din_fig2.png’)
#
#
# BUFFER_SIZE = 60000
# BATCH_SIZE = 256
# EPOCHS = 50
#
#
#
# # Batch and shuffle the data
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#
# resize_x = 28
# resize_y = 28
# #====================================
#=======================================med mnist dataset husk å endre size_of_last_filter_in_generator til 1 og conv3_filters til 1


"""
The dataset is now ready for training

"""

def make_generator_model(tensor_size_x, tensor_size_y):
    """
    sett inn hardkodede tall for parametrene under trening for å unngå for mye beregninger.
    """

    #use_bias = False this is to reduce the models complexity
    #noise parameters
    tensor_size_x = tensor_size_x
    tensor_size_y = tensor_size_y
    depth_feature_map = 256*2
    noise_vector = 512

    #convolution parameter
    conv1_filters = 128*2 # A filter is a matrice of numbers
    conv1_kernel_size = (5,5) # the kernel size is the size of that filter.

    conv2_filters = 64*2
    conv2_kernel_size = (5, 5)

    conv3_filters = 32 *2
    conv3_kernel_size = (5,5)

    last_layer_filter = color_channel
    last_layer_kernel_size = (5,5)

    model = tf.keras.Sequential()
    model.add(layers.Dense(tensor_size_x * tensor_size_y * depth_feature_map, use_bias=True, input_shape=(noise_vector,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((tensor_size_x, tensor_size_y, depth_feature_map)))
    assert model.output_shape == (None, tensor_size_x, tensor_size_y, depth_feature_map)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(conv1_filters, conv1_kernel_size, strides=(1, 1), padding='same', use_bias=True))
    assert model.output_shape == (None, tensor_size_x, tensor_size_y, conv1_filters)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(conv2_filters, conv2_kernel_size, strides=(2, 2), padding='same', use_bias=True)) #filter reduce stride = 2
    assert model.output_shape == (None, tensor_size_x * 2, tensor_size_y * 2, conv1_filters / 2) #strides increase the size (14,14,64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #======================ekstra layer for increased image size======================================
    model.add(layers.Conv2DTranspose(conv3_filters, conv3_kernel_size, strides=(2, 2), padding='same', use_bias=True)) #filter reduce stride = 2
    assert model.output_shape == (None, tensor_size_x * 2*2, tensor_size_y * 2*2, conv1_filters / 4) #strides increase the size (14,14,64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #======================ekstra layer for increased image size======================================
    #husk å endre på size til input når det legges til et nytt lag.

    model.add(layers.Conv2DTranspose(last_layer_filter, last_layer_kernel_size, strides=(2, 2), padding='same', use_bias=True, activation='tanh')) # filter = 1 we want black white, rgb conv3 = 3
    assert model.output_shape == (None, tensor_size_x * 2 * 2 *2, tensor_size_y * 2 * 2 * 2, last_layer_filter) #a test that our image has the expected shape

    return model
def make_discriminator_model(input_x,input_y):

    size_of_input_image_x = input_x
    size_of_input_image_y = input_y
    size_of_last_filter_in_generator = color_channel

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

#region test generator and discriminator

generator = make_generator_model(resize_x//8,resize_y//8) #deler på 4 fordi vi har strides 2 to steder

noise = tf.random.normal([BATCH_SIZE, noise_vector_for_gen]) # kan økes fra 100 for å gi mer kompleksitet i trening, men vil kreve mer minne og beregning
generated_image = generator(noise, training=False)
print(f"generated image generator: {generated_image.shape}")
generated_image = (generated_image - np.min(generated_image)) / (np.max(generated_image) - np.min(generated_image))
plt.imshow(generated_image[0, :, :, :])#,cmap="gray")# cmap='gray'
plt.title("Generated image")
plt.show()


noise_output_shape = generated_image.shape
print(f"=====================================noise output shape: {noise_output_shape}========================================")
x_value_generated_image = noise_output_shape[1]
y_value_generated_image = noise_output_shape[2]

discriminator = make_discriminator_model(x_value_generated_image, y_value_generated_image)
decision = discriminator(generated_image)
print(f"If the value is positive the image is real {decision}")

#endregion

"""
Define loss and optimizer functions
"""
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output) #Generator loss bruker barte fake output



generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# checkpoint_dir = "/media/joakim/Random Stuff/UbuntuFiles/pythonFiles/chk/masteroppgave_data_fra_trening"#'./training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
noise_dim = noise_vector_for_gen#200
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

"""
Her trenes både generator og discriminator
training = True
"""
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True) # fra dataset
      fake_output = discriminator(generated_images, training=True) #bilde generert av generator

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_loss', gen_loss, step=generator_optimizer.iterations)
        tf.summary.scalar('disc_loss', disc_loss, step=discriminator_optimizer.iterations)


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  #print(f"predictions.shape", predictions.shape)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      image = predictions[i]
      image = (image-np.min(image))/(np.max(image)-np.min(image))
      plt.imshow(image)#(predictions[i, :, :, :] * 127.5) + 127.5)#, cmap='hot') #kommentere ut cmap=gray???
      plt.axis('off')
  plt.suptitle(image_type)


  folder_name = 'generated_images_vanilla_GAN'
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)


  if epoch % 10 == 0:
      plt.savefig(os.path.join(folder_name, ' image_at_epoch_{:04d}.png'.format(epoch)))

      ##############################################################################################################
      #                         to neptune
        ##############################################################################################################

      image_path_buffer = f'{folder_name}/ image_at_epoch_{epoch:04d}.png'
      run[f"visualizations/from_training/image_at_epoch_{epoch:04d}"].upload(image_path_buffer)

      #run[f"visualizations/from_training/test_image_at_step_{epoch_num:04d}"].upload(image_path_buffer)

      print('fig closed')
      plt.close("all")
  #plt.show() plot for hver epoch
  #plt.savefig(‘din_fig.png’)


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    #if (epoch + 1) % 15 == 0:
     # checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start), " number of epochs: ", EPOCHS)

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

train(train_dataset, EPOCHS)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

end_time = time.time()  # Lagrer slutttiden
elapsed_time = end_time - start_time  # Beregner tiden det tok å kjøre koden

print(f"Tiden det tok å kjøre koden: {elapsed_time/60} minutter")

generator.save(f'saved_model_vanilla_GAN/{image_type[1:-8]}/my_generator.h5')
discriminator.save(f'saved_model_vanilla_GAN/{image_type[1:-8]}/my_discriminator.h5')
#print(f"image_type[1:]: {image_type[1:-8]}")

#
# # Display a single image using the epoch number (display as gif)
# def display_image(epoch_no):
#   return PIL.Image.open('generated_images/image_at_epoch_{:04d}.png'.format(epoch_no))
#
# display_image(EPOCHS)

run.stop()

time.sleep(30)  # for at slurm i fox ikke skal avslutte jobben før neptune har gjort seg ferdig
print("End of line")

