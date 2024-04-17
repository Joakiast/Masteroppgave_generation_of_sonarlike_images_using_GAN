fimport tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
@tf.keras.utils.register_keras_serializable(package='Custom', name='InstanceNormalization')
class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=input_shape[-1:], initializer='ones', trainable=True)
        self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        return self.scale * (inputs - mean) / tf.sqrt(variance + self.epsilon) + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

# Last inn modellen som før
generator = load_model('saved_model_cycle_GAN/cycleGAN_272/272_generator_g.h5')



def load_and_preprocess_image_simulated_set(path_simulated_image_trainset):
    # path_simulated_image_trainset = pathlib.Path("datasets")
    if isinstance(path_simulated_image_trainset, tf.Tensor):
        # print("===================start load and preprocess image============================================")

        inp_image = tf.io.read_file(path_simulated_image_trainset)
        inp_image = tf.image.decode_png(inp_image,
                                        channels=3)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde

        inp_image = tf.cast(inp_image, tf.float32)
        inp_image = (inp_image - 127.5) / 127.5
        # if not "clutter" in image_type:
        inp_image.set_shape([369, 496, 3])
        inp_image = tf.image.resize(inp_image, [256, 256], method=tf.image.ResizeMethod.LANCZOS5)
        return inp_image
    else:
        # print("===================start load and preprocess image============================================")

        # =================================for inp image======================================================
        inp_image = tf.io.read_file(path_simulated_image_trainset)
        inp_image = tf.image.decode_png(inp_image, channels=3)
        # print(f" inp_image.shape: {inp_image.shape}")
        inp_image = tf.cast(inp_image, tf.float32)
        inp_image = (inp_image - 127.5) / 127.5
        assert inp_image.shape == (369, 496, 3)
        inp_image = tf.image.resize(inp_image, [256, 256], method=tf.image.ResizeMethod.LANCZOS5)

        return inp_image

def save_label_file(label_folder, filename, content):
    label_file_path = os.path.join(label_folder, f"{filename[:-4]}.txt")  # Fjern '.png' og legg til '.txt'
    with open(label_file_path, 'w') as file:
        file.write(content)
    print(f"Label fil lagret: {label_file_path}")


# inp_image_path = 'simulerte_bilder/test_set_cycleGAN/barrel4_20_3.1676888064907254_9.0_0_70.png'  # Erstatt med stien til ditt bilde
# inp = load_and_preprocess_image_simulated_set(inp_image_path)
#
# # Siden modellen forventer en batch av bilder, legg til en batch-dimensjon
# inp = tf.expand_dims(inp, 0)  # inp.shape blir nå (1, 256, 256, 3) for eksempel
#
#
#
# generated_image = generator.predict(inp)
#
#
# if generated_image.ndim == 4:
#     generated_image = np.squeeze(generated_image, axis=0)
#
# # Bruk plt.imshow for å vise bildet.
# plt.imshow(inp[0])
# plt.title("input Image")
# plt.axis('off')  # Skjul aksemarkørene.
# plt.show()
#
# generated_image_clipped = np.clip(generated_image * 0.5 + 0.5, 0, 1)
#
# plt.imshow(generated_image_clipped)
# plt.title("Generated Image")
# plt.axis('off')
# plt.show()
#
# save_path = "generated_data/generated_image1.png"
#
# plt.imsave(save_path, generated_image_clipped)


input_folder_path = 'simulerte_bilder/test_set_cycleGAN'

# Sti til mappen hvor du vil lagre de genererte bildene
output_folder_path = 'generated_data'

# Sjekker og oppretter ut-mappen hvis den ikke eksisterer
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Gå gjennom hver fil i input-mappen
for filename in os.listdir(input_folder_path):
    if filename.lower().endswith('.png'):  # Behandler kun PNG-bilder
        file_path = os.path.join(input_folder_path, filename)
        inp = load_and_preprocess_image_simulated_set(file_path)

        # Legg til en batch-dimensjon
        inp_batch = tf.expand_dims(inp, 0)

        # Generer bildet
        generated_image = generator.predict(inp_batch)

        if generated_image.ndim == 4:
            generated_image = np.squeeze(generated_image, axis=0)  # Fjern batch-dimensjonen

        generated_image_clipped = np.clip(generated_image * 0.5 + 0.5, 0, 1)  # Klipp til gyldige verdier

        # Bygg filnavnet for det genererte bildet og lagre det
        save_path = os.path.join(output_folder_path, f"generated_oil_drum_RGB_{filename}")
        plt.imsave(save_path, generated_image_clipped)
        print(f"Bilde lagret: {save_path}")

        # Opprett en label-mappe hvis den ikke finnes
        label_folder_path = os.path.join(output_folder_path, "Label")
        if not os.path.exists(label_folder_path):
            os.makedirs(label_folder_path)

        # Lagre tilsvarende label-fil
        save_label_file(label_folder_path, f"generated_oil_drum_RGB_{filename}", "oil_drum 128 128")
