
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

generator = tf.keras.models.load_model('random/saved_model_impaining/saved_model_pix2pix_inpainting/oil_/my_generator.h5')
discriminator = tf.keras.models.load_model('random/saved_model_impaining/saved_model_pix2pix_inpainting/oil_/my_discriminator.h5')

def load_and_prepare_image(image_path, img_height=256, img_width=256):
    # Last inn bildet
    img = tf.io.read_file(image_path)
    # Dekoder bildet til en tensor og sikre at den har 3 kanaler (for RGB)
    img = tf.image.decode_jpeg(img, channels=3)
    # Endre størrelsen på bildet til ønsket størrelse
    img = tf.image.resize(img, [img_height, img_width])
    # Skaler pikselverdiene til området [-1, 1] som forventet av modellen din
    img = (img - 127.5) / 127.5
    return img



inp_image_path = 'image_translation_handdrawn_images/Screenshot from 2024-02-11 17-13-58.png'  # Erstatt med stien til ditt bilde
inp = load_and_prepare_image(inp_image_path)

# Siden modellen forventer en batch av bilder, legg til en batch-dimensjon
inp = tf.expand_dims(inp, 0)  # inp.shape blir nå (1, 256, 256, 3) for eksempel



generated_image = generator.predict(inp)

generated_image = (generated_image[0] + 1) / 2.0  # Reskalerer fra [-1, 1] til [0, 1]

# Kontroller at bildet ikke har en ekstra batch-dimensjon som kan forstyrre visningen.
if generated_image.ndim == 4:
    generated_image = np.squeeze(generated_image, axis=0)

# Bruk plt.imshow for å vise bildet.
plt.imshow(generated_image)
plt.title("Generated Image")
plt.axis('off')  # Skjul aksemarkørene.
plt.show()