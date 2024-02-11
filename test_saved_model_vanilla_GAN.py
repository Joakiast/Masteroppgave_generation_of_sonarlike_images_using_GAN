import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

generator = tf.keras.models.load_model('saved_model_vanilla_GAN/oil_/my_generator.h5')
discriminator = tf.keras.models.load_model('saved_model_vanilla_GAN/oil_/my_discriminator.h5')

num_examples_to_generate=16
noise_dim = 512
# Generere støyinput
noise = tf.random.normal([num_examples_to_generate, noise_dim])
generated_images = generator(noise, training=False)

 #Normaliser bildene en gang for alle bilder
generated_images = (generated_images - np.min(generated_images)) / (np.max(generated_images) - np.min(generated_images))

# Vis de genererte bildene
plt.figure(figsize=(10,10))  # Øker størrelsen på figuren for bedre visning
for i in range(num_examples_to_generate):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i, :, :, :])
    plt.axis('off')
plt.show()

plt.figure(figsize=(10,10))  # Øker størrelsen på figuren for bedre visning

plt.imshow(generated_images[1, :, :, :])
plt.axis('off')
plt.show()