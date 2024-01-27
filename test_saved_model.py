import tensorflow as tf
import matplotlib.pyplot as plt

generator = tf.keras.models.load_model('my_generator_model_oil_drum.h5')
discriminator = tf.keras.models.load_model('my_discriminator_model_oildrum_800epoch.h5')

num_examples_to_generate=16
noise_dim = 200
# Generere støyinput
noise = tf.random.normal([num_examples_to_generate, noise_dim])
generated_images = generator(noise, training=False)

# Vis de genererte bildene
for i in range(num_examples_to_generate):
    plt.subplot(4, 4, i+1)
    plt.imshow((generated_images[i, :, :, 0] * 127.5 + 127.5).numpy().astype('uint8'))
    plt.axis('off')
plt.show()


# Vis det genererte bildet
plt.imshow((generated_images[0, :, :, 0] * 127.5 + 127.5).numpy().astype('uint8'))
plt.axis('off')
plt.show()
