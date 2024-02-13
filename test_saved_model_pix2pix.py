
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

generator = tf.keras.models.load_model('random/sample2/saved_model_pix2pix_inpainting/oil_/my_generator.h5')
discriminator = tf.keras.models.load_model('random/sample2/saved_model_pix2pix_inpainting/oil_/my_discriminator.h5')

def load_and_prepare_image(image_path, img_height=256, img_width=256):
    # Last inn bildet
    img = tf.io.read_file(image_path)
    # Dekoder bildet til en tensor og sikre at den har 3 kanaler (for RGB)
    img = tf.image.decode_jpeg(img, channels=3)
    # Endre størrelsen på bildet til ønsket størrelse
    img = tf.image.resize(img, [img_height, img_width])
    # Skaler pikselverdiene til området [-1, 1] som forventet av modellen din
    #img = (img - 127.5) / 127.5
    return img



inp_image_path = 'random/sample2/saved_model_pix2pix_inpainting/oil_/Screenshot 2024-02-12 161658.png'  # Erstatt med stien til ditt bilde
inp = load_and_prepare_image(inp_image_path)

# Siden modellen forventer en batch av bilder, legg til en batch-dimensjon
inp = tf.expand_dims(inp, 0)  # inp.shape blir nå (1, 256, 256, 3) for eksempel


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




generator = tf.keras.models.load_model(
    'random/sample2/saved_model_pix2pix_inpainting/oil_/my_generator.h5',
    custom_objects={'downsample': downsample, 'upsample': upsample}
)

generated_image = generator.predict(inp)

#generated_image = (generated_image[0] + 1) / 2.0  # Reskalerer fra [-1, 1] til [0, 1]

# Kontroller at bildet ikke har en ekstra batch-dimensjon som kan forstyrre visningen.
if generated_image.ndim == 4:
    generated_image = np.squeeze(generated_image, axis=0)

# Bruk plt.imshow for å vise bildet.
plt.imshow("random/sample2/saved_model_pix2pix_inpainting/oil_/Screenshot 2024-02-12 161658.png")
plt.title("Generated Image")
plt.axis('off')  # Skjul aksemarkørene.
plt.show()







