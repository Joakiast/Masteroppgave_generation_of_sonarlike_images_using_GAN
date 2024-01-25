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


start_time = time.time()
# Sti til mappen der bildene dine er plassert
train_set_path = pathlib.Path("train")

"""
Dersom jeg ønsker rock så kommenter ut de 2 andre 
"""
BATCH_SIZE = 100
image_type = '*rock_RGB.jpg'
#image_type = '*oil_drum_RGB.jpg'
#image_type = '*clutter_RGB.jpg'


# Opprett en liste over bildestier som strenger
image_paths = [str(path) for path in list(train_set_path.glob(image_type))]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of trainingset: {len(image_paths)}")
# Funksjon for å lese og forbehandle bildene
resize_x = 100
resize_y = 100

"""
increase the dataset used for "rock and oil" 
"""

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # Bruk tf.image.decode_png for PNG-bilder, etc.
    image = tf.image.resize(image, [resize_y, resize_x],method=tf.image.ResizeMethod.AREA) #ønsket resize størrelse, jo mindre jo raskere og dårligere kvalitet
    image = tf.cast(image, tf.float32)
    print(f"image shape: {image.shape}")
    image = (image - 127.5) /127.5  # Normaliser bildene til [-1, 1] området
    return image

BUFFER_SIZE = len(image_paths)

EPOCHS = 100
#print(BUFFER_SIZE)
flipped_images_left_to_right = []  # Opprett en liste for de augmenterte bildene
flipped_images_up_down = []
for image_path in image_paths:
    original_image = load_and_preprocess_image(image_path)
    if "rock_RGB" in image_type:
        #print("Det er rock, gjør noe")
        flip_image_left_right = tf.image.flip_left_right(load_and_preprocess_image(image_path))
        flip_image_up_down = tf.image.flip_up_down(load_and_preprocess_image(image_path))
    elif "oil_drum_RGB" in image_type:
        #print("Det er oil, gjør noe")
        flip_image_left_right = load_and_preprocess_image(image_path)
        flip_image_up_down = tf.image.flip_up_down(load_and_preprocess_image(image_path))

    else:
        #print("Det er clutter, gjør noe annet")
        flip_image_left_right = None #load_and_preprocess_image(image_path)
        flip_image_up_down = None

    flipped_images_left_to_right.append(flip_image_left_right)
    flipped_images_up_down.append(flip_image_up_down)
print(f"augmented_images.shape {len(flipped_images_left_to_right)}")

# Opprett en tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
print(f"dataset shape 1: {len(train_dataset)}")
train_dataset = train_dataset.map(load_and_preprocess_image)
print(f"dataset shape 2: {len(train_dataset)}")
flipped_images_left_right = tf.data.Dataset.from_tensor_slices(flipped_images_left_to_right)
train_dataset = train_dataset.concatenate(flipped_images_left_right)

print(f"dataset shape 3: {len(train_dataset)}")
flipped_images_up_down = tf.data.Dataset.from_tensor_slices(flipped_images_up_down)
train_dataset = train_dataset.concatenate(flipped_images_up_down)
print(f"dataset shape 4: {len(train_dataset)}")

train_dataset = train_dataset.shuffle(BUFFER_SIZE)  # Bland datasettet, hvis ønskelig
print(f"dataset shape 5: {len(train_dataset)}")
train_dataset = train_dataset.batch(BATCH_SIZE)  # Velg en batch-størrelse som passer for din maskin
print(f"dataset shape 6: {len(train_dataset)}")
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # For ytelsesoptimalisering
print(f"dataset shape 7: {len(train_dataset)}")

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

    model.add(layers.Conv2DTranspose(conv3_filters, conv3_kernel_size, strides=(2, 2), padding='same', use_bias=True, activation='tanh')) # filter = 1 we want black white, rgb conv3 = 3
    assert model.output_shape == (None, tensor_size_x * 2 * 2, tensor_size_y * 2 * 2, conv3_filters) #a test that our image has the expected shape

    return model
def make_discriminator_model(input_x,input_y):

    size_of_input_image_x = input_x
    size_of_input_image_y = input_y
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

#region test generator and discriminator

generator = make_generator_model(resize_x//4,resize_y//4) #deler på 4 fordi vi har strides 2 to steder

noise = tf.random.normal([1, 100]) # kan økes fra 100 for å gi mer kompleksitet i trening, men vil kreve mer minne og beregning
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])#,cmap="gray")# cmap='gray'
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

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
noise_dim = 100
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

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  #print(f"predictions.shape", predictions.shape)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow((predictions[i, :, :, 0] * 127.5) + 127.5)#, cmap='rgb') #kommentere ut cmap=gray???
      plt.axis('off')

  folder_name = 'generated_images'
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)


  plt.savefig(os.path.join(folder_name,'image_at_epoch_{:04d}.png'.format(epoch)))
  if epoch % 10 == 0:
      plt.savefig(os.path.join(folder_name, 'image_at_epoch_{:04d}.png'.format(epoch)))
      #plt.close(fig)
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
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

train(train_dataset, EPOCHS)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

end_time = time.time()  # Lagrer slutttiden
elapsed_time = end_time - start_time  # Beregner tiden det tok å kjøre koden

print(f"Tiden det tok å kjøre koden: {elapsed_time} sekunder")

# Display a single image using the epoch number (display as gif)
def display_image(epoch_no):
  return PIL.Image.open('generated_images/image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)