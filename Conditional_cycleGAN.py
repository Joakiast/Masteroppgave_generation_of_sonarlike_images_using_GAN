"""
The CycleGAN paper uses a modified resnet
based generator. This tutorial is using a modified unet generator for simplicity.
"""

# ==============================================================================

import tensorflow as tf
from keras.src.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

tf.__version__
import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
import PIL
from tensorflow.keras import layers
import time
import pathlib
import tensorflow_addons as tfa
from IPython import display
from IPython.display import clear_output
import neptune
from io import BytesIO
from neptune.types import File
import datetime
import sys
# import cv2
# from sklearn.cluster import KMeans
import math
import tensorflow_addons as tfa

from tensorflow.keras.utils import register_keras_serializable



from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import random

from tensorflow_addons.layers import SpectralNormalization

#from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import peak_signal_noise_ratio as psnr


# Set seeds for reproducibility
seed_number = 42

tf.random.set_seed(seed_number)
np.random.seed(seed_number)
random.seed(seed_number)

run = neptune.init_run(
    project="masteroppgave/ConditionalCycleGAN",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMDY3ZDFlNS1hMGVhLTQ1N2YtODg4MC1hNThiOTM1NGM3YTQifQ=="
)

# def upload_plot_to_neptune(imgs, titles, contrast=8):
#     plt.figure(figsize=(8, 8))
#     for i in range(len(imgs)):
#         plt.subplot(2, 2, i+1)
#         plt.title(titles[i])
#         if i % 2 == 0:
#             plt.imshow(imgs[i][0] * 0.5 + 0.5)
#         else:
#             plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
#         plt.axis('off')
#
#     # Lagre til en BytesIO stream og laste opp direkte til Neptune
#     img_buffer = BytesIO()
#     plt.savefig(img_buffer, format='png')
#     img_buffer.seek(0)
#     run["visualizations/generator_outputs"].upload(img_buffer)
#     plt.close()


start_time = time.time()
# region load the dataset
# test 0
resize_x = 256
resize_y = 256

# The bath size of 1 gives better results using the UNet in this experiment.
BATCH_SIZE = 1
BATCH_SIZE_TEST = 1  # BATCH_SIZE
EPOCHS = 2
decay_start_epoch = 100
color_channel = 3
crop_size = 256  # resize_x / 2 150 fin størrelse på
DROPOUT = 0  # .5
LAMBDA = 10

learningrate_G_g = 0.0002  # 7e-5
learningrate_G_f = 0.0002  # 7e-5
learningrate_D_x = learningrate_G_g / 2  # 4e-5
learningrate_D_y = learningrate_G_f / 2  # 4e-5

beta_G_g = 0.9
beta_G_f = 0.9
beta_D_x = 0.9
beta_D_y = 0.9

save_every_n_epochs = 2

# generator_type = "resnet" # virker som resnet gir best resultat
generator_type = "unet"

filter_muultiplier_generator = 2
filter_muultiplier_discriminator = 1

# image_type = '*rock_RGB'
image_type = '*oil_drum_RGB'
# image_type = '*clutter_RGB'
# image_type = "*man_made_object_RGB"
# image_type = "*Mine_size_rock_RGB.jpg"

image_type_2 = False
# image_type_2 = '*rock_RGB'
# image_type_2 = '*oil_drum_RGB'
# image_type_2 = "*man_made_object_RGB"
# image_type_2 = "*Mine_size_rock_RGB.jpg"


image_type_3 = False
# image_type_3 = '*rock_RGB'
# image_type_3 = '*oil_drum_RGB'
# image_type_3 = "*man_made_object_RGB"
# image_type = "*Mine_size_rock_RGB.jpg"


params = {
    "activation": "tanh",
    "n_epochs": EPOCHS,
    "decay_start_epoch": decay_start_epoch,
    "batch_size": BATCH_SIZE,
    "drop_out": DROPOUT,
    "learningrate_generator_g": learningrate_G_g,
    "learningrate_generator_f": learningrate_G_f,
    "learningrate_discriminator_x": learningrate_D_x,
    "learningrate_discriminator_y": learningrate_D_y,
    "beta_G_g": beta_G_g,
    "beta_G_f": beta_G_f,
    "beta_D_x": beta_D_x,
    "beta_D_y": beta_D_y,
    "Lambda": LAMBDA,
    "Image_type added with extra set": image_type,
    "use_bias": False,
    #  "number_of_filters": "increased x2 in generator not discriminator",
    "type of generator": generator_type,
    "type of loss func": "MeanSquaredError",
    "save_every_n_epochs": save_every_n_epochs,
    "filter multiplier gen": filter_muultiplier_generator,
    "filter multiplier disc": filter_muultiplier_discriminator,
    "seed_number": seed_number
}

if image_type_2:
    params["image_type_2"] = image_type_2

if image_type_3:
    params["image_type_3"] = image_type_3

run["model/parameters"] = params

# region Preparing datasets

train_set_path = pathlib.Path("datasets/train")
train_set_path_simulated = pathlib.Path(
    "datasets/barrel_sim_v2/sim_data_rgb_barrel_v2")  # ("datasets/sim_data_rgb_barrel") kommentert ut gammel simulert datasett
# train_set_path_simulated_v1 = pathlib.Path("datasets/sim_data_rgb_barrel")
test_set_path_simulated = pathlib.Path("datasets/test_set_cycleGAN")
# test_set_path = pathlib.Path("datasets/test")
# test_set_path_handdrawn = pathlib.Path("datasets/image_translation_handdrawn_images")
train_set_extra_path = pathlib.Path("datasets/test")

image_paths_train = [str(path) for path in list(
    train_set_path.glob(image_type + ".jpg"))]  # [:8000]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of trainingset: {len(image_paths_train)}")

image_paths_train_extra = [str(path) for path in list(
    train_set_extra_path.glob(
        image_type + ".jpg"))]  # [:8000]  # filterer ut data i datasettet i terminal: ls |grep oil

# image_paths_train_sim_V1 = [str(path) for path in list(train_set_path_simulated_v1.glob("*.png"))]
# print(f"size of simulert trainingset V1: {len(image_paths_train_sim_V1)}")

image_paths_train.extend(image_paths_train_extra)
print(f"size of sonar trainingset after adding extra sonar training data: {len(image_paths_train)}")

if image_type_2:
    img_buffer_1 = [str(path) for path in list(train_set_path.glob(image_type_2 + ".jpg"))]  # [:8000]
    image_paths_train.extend(img_buffer_1)
if image_type_3:
    img_buffer_2 = [str(path) for path in list(train_set_path.glob(image_type_3 + ".jpg"))]  # [:8000]
    image_paths_train.extend(img_buffer_2)

image_paths_train_simulated = [str(path) for path in list(train_set_path_simulated.glob("*.png"))][
                              :425]  # total størrelse 425   # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of simulated trainingset:: {len(image_paths_train_simulated)}")
# image_paths_train_simulated.extend(image_paths_train_sim_V1)
# print(f"size of simulated trainingset after adding extra simulated data: {len(image_paths_train_simulated)}")

image_paths_test = [str(path) for path in list(test_set_path_simulated.glob(
    "*.png"))]  # [str(path) for path in list(train_set_path_simulated.glob("*.png"))][553:]  # filterer ut data i datasettet i terminal: ls |grep oil
print(f"size of testset: {len(image_paths_test)}")


# buffer_test = [str(path) for path in list(test_set_path_handdrawn.glob(
#   "*.png"))]  # [:405] #total størrelse 425   # filterer ut data i datasettet i terminal: ls |grep oil
# image_paths_test.extend(buffer_test)


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

    # print(f"crop by {crop_size}")
    return image


def find_coordinates_for_cropping_tensor(path_image):
    # base_name = str[path_image]#tf.cast(path_image, str)
    base_name_b = os.path.basename(path_image.numpy())
    base_name = base_name_b.decode("utf-8")
    # print(f"base name {base_name}")
    label_file = base_name.replace('.jpg', '.txt')  # Bytt ut filendelsen fra .jpg til .txt
    # print(f"label file {label_file}")

    if hasattr(path_image, 'numpy'):
        path_image_str = path_image.numpy().decode("utf-8")
    else:
        path_image_str = path_image

    #####################
    if "test" in path_image_str:
        label_path = os.path.join("datasets/test/Label", label_file)
    else:
        label_path = os.path.join("datasets/train/Label", label_file)
    #####################

    # label_path = os.path.join("datasets/train/Label", label_file)
    # print(f"label_path {label_path}")
    x, y = None, None
    try:

        with open(label_path, 'r') as file:
            label_content = file.read()

        for line in label_content.split('\n'):
            parts = line.split()
            # if parts and parts[0] != 'clutter':
            x, y = map(float, parts[1:3])
            # print(f"x: {x}, y: {y}")
            return x, y

    except Exception as e:
        print(f"Error while processing label file {label_path}: {e}")
    return None, None


def find_coordinates_for_cropping(path_image):
    base_name = os.path.basename(path_image)  # .numpy())
    # print(f"base name {base_name}")
    label_file = base_name.replace('.jpg', '.txt')  # Bytt ut filendelsen fra .jpg til .txt
    # print(f"label file {label_file}")

    if hasattr(path_image, 'numpy'):
        path_image_str = path_image.numpy().decode("utf-8")
    else:
        path_image_str = path_image

    ####################
    if "test" in path_image_str:
        label_path = os.path.join("datasets/test/Label", label_file)
    else:
        label_path = os.path.join("datasets/train/Label", label_file)
    #####################

    # label_path = os.path.join("datasets/train/Label", label_file)
    # print(f"label_path {label_path}")
    x, y = None, None
    try:

        with open(label_path, 'r') as file:
            label_content = file.read()

        for line in label_content.split('\n'):
            parts = line.split()
            # if parts and parts[0] != 'clutter':
            x, y = map(float, parts[1:3])
            # print(f"x: {x}, y: {y}")
            return x, y
            # elif parts and parts[0] == 'rock':
            #     x, y = map(float, parts[1:3])
            #     return x,y

    except Exception as e:
        print(f"Error while processing label file {label_path}: {e}")
    return None, None


def packout_strings(paths):
    inp, re = paths
    return inp, re


# ==========================
def load_and_preprocess_image_trainset(paths):
    # path_simulated_image_trainset = pathlib.Path("datasets")
    if isinstance(paths, tf.Tensor):
        # path_image_trainset, path_simulated_image_trainset = paths
        path_image_trainset, path_simulated_image_trainset = tf.py_function(func=packout_strings, inp=[paths],
                                                                            Tout=[tf.string, tf.string])
        # print("===================start load and preprocess image============================================")
        real_image = tf.io.read_file(path_image_trainset)
        inp_image = tf.io.read_file(path_simulated_image_trainset)
        real_image = tf.image.decode_jpeg(real_image,
                                          channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
        inp_image = tf.image.decode_png(inp_image, channels=color_channel)
        real_image = tf.cast(real_image, tf.float32)
        inp_image = tf.cast(inp_image, tf.float32)
        real_image = (real_image - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
        inp_image = (inp_image - 127.5) / 127.5
        x, y = tf.py_function(func=find_coordinates_for_cropping_tensor, inp=[path_image_trainset],
                              Tout=[tf.float32, tf.float32])
        real_image.set_shape([400, 600, 3])
        inp_image.set_shape([369, 496, 3])
        real_image = crop_image_around_POI(real_image, x, y, crop_size)

        real_image = tf.image.resize(real_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
        inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
        # ====================for inp===============================================================================

        return inp_image, real_image
    else:
        path_image_trainset, path_simulated_image_trainset = paths
        # print("===================start load and preprocess image============================================")
        real_image = tf.io.read_file(path_image_trainset)
        inp_image = tf.io.read_file(path_simulated_image_trainset)
        real_image = tf.image.decode_jpeg(real_image,
                                          channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
        inp_image = tf.image.decode_png(inp_image, channels=color_channel)
        real_image = tf.cast(real_image, tf.float32)
        inp_image = tf.cast(inp_image, tf.float32)
        real_image = (real_image - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
        inp_image = (inp_image - 127.5) / 127.5
        assert real_image.shape == (400, 600, 3)
        assert inp_image.shape == (369, 496, 3)
        # image = tf.image.resize(image, [400, 600], method=tf.image.ResizeMethod.AREA)
        # if not "clutter" in image_type:
        x, y = find_coordinates_for_cropping(path_image_trainset)
        real_image = crop_image_around_POI(real_image, x, y, crop_size)
        # image = tf.image.resize(image, [resize_x,resize_y], method=tf.image.ResizeMethod.AREA)
        # print(f"alle bilder kommer hit: image shape før resize: {image.shape} bilde: {path_image}")
        real_image = tf.image.resize(real_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
        inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)

        return inp_image, real_image


def load_and_preprocess_image_simulated_set(paths):
    # path_simulated_image_trainset = pathlib.Path("datasets")
    if isinstance(paths, tf.Tensor):

        path_simulated_image_trainset, path_image_trainset = tf.py_function(func=packout_strings, inp=[paths],
                                                                            Tout=[tf.string, tf.string])

        # print("===================start load and preprocess image============================================")
        inp_image = tf.io.read_file(path_simulated_image_trainset)
        real_image = tf.io.read_file(path_image_trainset)
        inp_image = tf.image.decode_png(inp_image,
                                        channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
        real_image = tf.image.decode_jpeg(real_image, channels=color_channel)
        inp_image = tf.cast(inp_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)
        inp_image = (inp_image - 127.5) / 127.5
        real_image = (real_image - 127.5) / 127.5
        x, y = tf.py_function(func=find_coordinates_for_cropping_tensor, inp=[path_image_trainset],
                              Tout=[tf.float32, tf.float32])
        real_image.set_shape([400, 600, 3])
        # if not "clutter" in image_type:
        inp_image.set_shape([369, 496, 3])
        real_image = crop_image_around_POI(real_image, x, y, crop_size)
        inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
        real_image = tf.image.resize(real_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
        return real_image, inp_image
    else:
        path_simulated_image_trainset, path_image_trainset = paths

        # print("===================start load and preprocess image============================================")

        # =================================for inp image======================================================
        inp_image = tf.io.read_file(path_simulated_image_trainset)
        real_image = tf.io.read_file(path_image_trainset)
        inp_image = tf.image.decode_png(inp_image, channels=color_channel)
        real_image = tf.image.decode_jpeg(real_image, channels=color_channel)
        # print(f" inp_image.shape: {inp_image.shape}")
        inp_image = tf.cast(inp_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)
        inp_image = (inp_image - 127.5) / 127.5
        real_image = (real_image - 127.5) / 127.5
        x, y = find_coordinates_for_cropping(path_image_trainset)
        real_image.set_shape([400, 600, 3])
        assert inp_image.shape == (369, 496, 3)
        assert real_image.shape == (400, 600, 3)
        real_image = crop_image_around_POI(real_image, x, y, crop_size)
        inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
        real_image = tf.image.resize(real_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)

        return real_image, inp_image


def load_and_preprocess_image_test_set(path_simulated_image_trainset):
    # path_simulated_image_trainset = pathlib.Path("datasets")
    if isinstance(path_simulated_image_trainset, tf.Tensor):
        # print("===================start load and preprocess image============================================")

        inp_image = tf.io.read_file(path_simulated_image_trainset)
        inp_image = tf.image.decode_png(inp_image,
                                        channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde

        inp_image = tf.cast(inp_image, tf.float32)
        inp_image = (inp_image - 127.5) / 127.5
        # if not "clutter" in image_type:
        inp_image.set_shape([369, 496, 3])
        inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
        return inp_image
    else:
        # print("===================start load and preprocess image============================================")

        # =================================for inp image======================================================
        inp_image = tf.io.read_file(path_simulated_image_trainset)
        inp_image = tf.image.decode_png(inp_image, channels=color_channel)
        # print(f" inp_image.shape: {inp_image.shape}")
        inp_image = tf.cast(inp_image, tf.float32)
        inp_image = (inp_image - 127.5) / 127.5
        assert inp_image.shape == (369, 496, 3)
        inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)

        return inp_image


BUFFER_SIZE_trainset = len(image_paths_train)
print(f"BUFFER_SIZE train set:: {BUFFER_SIZE_trainset}")

BUFFER_SIZE_simulated = len(image_paths_train_simulated)
print(f"BUFFER_SIZE simulated set: {BUFFER_SIZE_simulated}")

BUFFER_SIZE_test_set = len(image_paths_test)


def augmentation(inp, re):
    # print(f"input in augmentation len: {len(inp)}")
    # real_img, inp_img, = inp
    # print(f"real_img shape: {real_img.shape} inp_img shape: {inp_img.shape}")

    flipped_left_right = (tf.image.flip_left_right(re), tf.image.flip_left_right(inp))
    flipped_up_down = (tf.image.flip_up_down(re), tf.image.flip_up_down(inp))
    n_degrees = 10  # degrees rotated
    radians = n_degrees * math.pi / 180
    rotate = (tfa.image.rotate(re, -radians), tfa.image.rotate(inp, -radians))  # tfa.image.rotate(inp,-radians)

    # rotate = tf.image.rot90(input_img)
    # flytte objektet til forksjellige posisjoner i bildet, være forskintg rotering av sonar bilder med tanke på skygge 10 grader opp og ned
    # kan også være interresant å se på ville augemnteringer som farge og store rotasjoner
    # kan dele det inn i fysiske realiserbare og ikke realiserbare
    # kan også være innteresant å croppe og resize så det ikke blir helt likt det blir da ikke fysisk realiserbart, men kan være av interesse
    # bør ta det til sist liten tro på
    # print(f"inne i augmentation flipped_left_right len: {len(flipped_left_right)} flipped_up_down len: {len(flipped_up_down)} rotate len: {len(rotate)}")

    return flipped_left_right, flipped_up_down, rotate


inp_augmented_training_data_flip_left_right = []
real_augmented_training_data_flip_left_right = []

inp_augmented_training_data_flip_up_down = []
real_augmented_training_data_flip_up_down = []

inp_augmented_training_data_rotate = []
real_augmented_training_data_rotate = []

inp_augmented_simulated_data_flip_left_right = []
real_augmented_simulated_data_flip_left_right = []

inp_augmented_simulated_data_flip_up_down = []
real_augmented_simulated_data_flip_up_down = []

inp_augmented_simulated_data_rotate = []
real_augmented_simulated_data_rotate = []

i = 0
for image_path, image_path_simulated in zip(image_paths_train, image_paths_train_simulated):
    re_x, inp_x = load_and_preprocess_image_trainset((image_path, image_path_simulated))
    re_y, inp_y = load_and_preprocess_image_simulated_set((image_path_simulated, image_path))

    # print(f"input_to_train_set len: {len(re_x)} re_to_trainset len: {len(inp_x)}")
    # print(f"input_to_train_set shape: {re.shape} re_to_trainset shape: {inp.shape}")

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
    # i += 1
    # if i > 9:
    #     sys.exit()
    # tf.py_function(func=find_coordinates_for_cropping_tensor, inp=[path_image], Tout=[tf.float32,tf.float32])
    # plt.show()
    # only augment the image if we dont have a image type of clutter
    if "rock_RGB" in image_type or "oil_drum_RGB" in image_type or "man_made_object_RGB" in image_type:
        # print(f"re shape: {re.shape}")
        train_flipped_left_right, train_flipped_up_down, train_rotate = augmentation(re_x, inp_x)

        flipped_left_right_train_re, flipped_left_right_train_inp = train_flipped_left_right
        real_augmented_training_data_flip_left_right.append(flipped_left_right_train_re)  # , flipped_up_down, rotate])
        inp_augmented_training_data_flip_left_right.append(flipped_left_right_train_inp)
        real_augmented_up_down, inp_augmented_up_down = train_flipped_up_down
        real_augmented_training_data_flip_up_down.append(real_augmented_up_down)
        inp_augmented_training_data_flip_up_down.append(inp_augmented_up_down)
        real_augmented_rotate, inp_augmented_rotate = train_rotate
        real_augmented_training_data_rotate.append(real_augmented_rotate)
        inp_augmented_training_data_rotate.append(inp_augmented_rotate)

    elif "clutter" in image_type:
        pass

    # ===================================for simulated data===================================
    simulated_flipped_left_right, simulated_flipped_up_down, simulated_rotate = augmentation(re_y, inp_y)

    flipped_left_right_simulated_re, flipped_left_right_simulated_inp = simulated_flipped_left_right
    real_augmented_simulated_data_flip_left_right.append(flipped_left_right_simulated_re)
    inp_augmented_simulated_data_flip_left_right.append(flipped_left_right_simulated_inp)

    flipped_up_down_simulated_re, flipped_up_down_simulated_inp = simulated_flipped_up_down
    real_augmented_simulated_data_flip_up_down.append(flipped_up_down_simulated_re)
    inp_augmented_simulated_data_flip_up_down.append(flipped_up_down_simulated_inp)

    rotate_simulated_re, rotate_simulated_inp = simulated_rotate
    real_augmented_simulated_data_rotate.append(rotate_simulated_re)
    inp_augmented_simulated_data_rotate.append(rotate_simulated_inp)

    # ===================================for simulated data===================================

# for image_path_test in image_paths_test:
#     re_test,inp_test = load_and_preprocess_image(image_path_test)


# ======================
# Opprett et tf.data.Dataset fra bildestier
# the dataset consist of both inp and re images.
# region oppretter dataset for trening og inp
data_pairs = list(zip(image_paths_train, image_paths_train_simulated))

train_dataset = tf.data.Dataset.from_tensor_slices(data_pairs)
print(image_paths_train[0])
print(f"dataset train shape 1: {len(train_dataset)}")
data_pairs_2 = list(zip(image_paths_train_simulated, image_paths_train))
simulated_dataset = tf.data.Dataset.from_tensor_slices(data_pairs_2)
print(f"dataset simulated shape 1: {len(simulated_dataset)}")

train_dataset = train_dataset.map(load_and_preprocess_image_trainset, num_parallel_calls=tf.data.AUTOTUNE)
print(f"dataset shape 2: {len(train_dataset)}")
simulated_dataset = simulated_dataset.map(load_and_preprocess_image_simulated_set, num_parallel_calls=tf.data.AUTOTUNE)
print(f"dataset simulated shape 2: {len(simulated_dataset)}")

augmented_training_dataset_flip_left_right = tf.data.Dataset.from_tensor_slices(
    (real_augmented_training_data_flip_left_right, inp_augmented_training_data_flip_left_right))
augmented_simulated_dataset_flip_left_right = tf.data.Dataset.from_tensor_slices(
    (inp_augmented_training_data_flip_left_right, real_augmented_training_data_flip_left_right))
train_dataset = train_dataset.concatenate(augmented_training_dataset_flip_left_right)
simulated_dataset = simulated_dataset.concatenate(augmented_simulated_dataset_flip_left_right)
print(f"dataset shape 2: {len(train_dataset)}")
print(f"dataset simulated shape 2: {len(simulated_dataset)}")

augmented_training_dataset_flip_up_down = tf.data.Dataset.from_tensor_slices(
    (real_augmented_training_data_flip_up_down, inp_augmented_training_data_flip_up_down))
augmented_simulated_dataset_flip_up_down = tf.data.Dataset.from_tensor_slices(
    (inp_augmented_training_data_flip_up_down, real_augmented_training_data_flip_up_down))
train_dataset = train_dataset.concatenate(augmented_training_dataset_flip_up_down)
simulated_dataset = simulated_dataset.concatenate(augmented_simulated_dataset_flip_up_down)
print(f"dataset shape 3: {len(train_dataset)}")
print(f"dataset simulated shape 3: {len(simulated_dataset)}")

augmented_training_dataset_rotate = tf.data.Dataset.from_tensor_slices(
    (real_augmented_training_data_rotate, inp_augmented_training_data_rotate))
augmented_simulated_dataset_rotate = tf.data.Dataset.from_tensor_slices(
    (inp_augmented_training_data_rotate, real_augmented_training_data_rotate))
train_dataset = train_dataset.concatenate(augmented_training_dataset_rotate)
simulated_dataset = simulated_dataset.concatenate(augmented_simulated_dataset_rotate)
print(f"dataset shape 4: {len(train_dataset)}")
print(f"dataset simulated shape 4: {len(simulated_dataset)}")

train_dataset = train_dataset.shuffle(BUFFER_SIZE_trainset)
simulated_dataset = simulated_dataset.shuffle(BUFFER_SIZE_simulated)
print(f"dataset shape 5: {len(train_dataset)}")
print(f"dataset simulated shape 5: {len(simulated_dataset)}")

train_dataset = train_dataset.batch(BATCH_SIZE)
simulated_dataset = simulated_dataset.batch(BATCH_SIZE)

print(f"dataset shape 6: {len(train_dataset)}")
print(f"dataset simulated shape 6: {len(simulated_dataset)}")
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
simulated_dataset = simulated_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(image_paths_test)
test_dataset = test_dataset.map(load_and_preprocess_image_test_set, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(BUFFER_SIZE_test_set)
test_dataset = test_dataset.batch(BATCH_SIZE_TEST)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# ============================================================================
# endregion
number_of_samples_to_show = BATCH_SIZE  # Antall eksempler du ønsker å vise

for i in train_dataset.take(1):
    print(f"Element type tuple len: {len(i)}")

# Tar en batch fra datasettet
for input_imgs, real_imgs in train_dataset.take(1):
    plt.figure(figsize=(10, 5))
    for i in range(number_of_samples_to_show):
        # Plotter input_img
        ax = plt.subplot(2, number_of_samples_to_show, 2 * i + 1)
        plt.title("Input Image")
        plt.imshow(input_imgs[i].numpy())

        # Plotter real_img
        ax = plt.subplot(2, number_of_samples_to_show, 2 * i + 2)
        plt.title("Real Image")
        plt.imshow(real_imgs[i].numpy())
        plt.axis('on')

plt.tight_layout()
plt.show()

sample_simulated = next(iter(simulated_dataset))
sample_train = next(iter(train_dataset))
sample_test = next(iter(test_dataset))

"""
Import and reuse the Pix2Pix models
Import the generator and the discriminator used in Pix2Pix via the installed tensorflow_examples package.

The model architecture used in this tutorial is very similar to what was used in pix2pix. Some of the differences are:

Cyclegan uses instance normalization instead of batch normalization.
The CycleGAN paper uses a modified resnet based generator. This tutorial is using a modified unet generator for simplicity.
"""
OUTPUT_CHANNELS = 3


# region Genrator and discriminator
# ===========================================================================================

@register_keras_serializable(package='Custom', name='InstanceNormalization')
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5, **kwargs):
        # Pass any additional keyword arguments to the superclass.
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """Downsamples an input.

    Conv2D => Batchnorm => LeakyRelu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_norm: If True, adds the batchnorm layer

    Returns:
      Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=True))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input.

    Conv2DTranspose => Batchnorm => Dropout => Relu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer

    Returns:
      Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=True))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(DROPOUT))

    result.add(tf.keras.layers.ReLU())

    return result


def unet_generator(filter_multiplier, output_channels, norm_type='batchnorm'):
    """Modified u-net generator model (https://arxiv.org/abs/1611.07004).

    Args:
      output_channels: Output channels
      norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

    Returns:
      Generator model
    """

    down_stack = [
        downsample(math.floor(64 * filter_multiplier), 4, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
        downsample(math.floor(128 * filter_multiplier), 4, norm_type),  # (bs, 64, 64, 128)
        downsample(math.floor(256 * filter_multiplier), 4, norm_type),  # (bs, 32, 32, 256)
        downsample(math.floor(512 * filter_multiplier), 4, norm_type),  # (bs, 16, 16, 512)
        downsample(math.floor(512 * filter_multiplier), 4, norm_type),  # (bs, 8, 8, 512)
        downsample(math.floor(512 * filter_multiplier), 4, norm_type),  # (bs, 4, 4, 512)
        downsample(math.floor(512 * filter_multiplier), 4, norm_type),  # (bs, 2, 2, 512)
        downsample(math.floor(512 * filter_multiplier), 4, norm_type)  # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(math.floor(512 * filter_multiplier), 4, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(math.floor(512 * filter_multiplier), 4, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(math.floor(512 * filter_multiplier), 4, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(math.floor(512 * filter_multiplier), 4, norm_type),  # (bs, 16, 16, 1024)
        upsample(math.floor(256 * filter_multiplier), 4, norm_type),  # (bs, 32, 32, 512)
        upsample(math.floor(128 * filter_multiplier), 4, norm_type),  # (bs, 64, 64, 256)
        upsample(math.floor(64 * filter_multiplier), 4, norm_type),  # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
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
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# =========================Resnet=====================================

def ResidualBlock(x, filters, size, norm_type='instancenorm', apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    conv_block = tf.keras.Sequential()
    conv_block.add(layers.Conv2D(filters, size, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
    if norm_type == 'instancenorm':
        conv_block.add(InstanceNormalization())
    elif norm_type == 'batchnorm':
        conv_block.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        conv_block.add(layers.Dropout(DROPOUT))
    conv_block.add(layers.ReLU())
    conv_block.add(layers.Conv2D(filters, size, strides=1, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
    if norm_type == 'instancenorm':
        conv_block.add(InstanceNormalization())

    return layers.add([x, conv_block(x)])


def ResNetGenerator(filter_multiplier, input_shape=(resize_x, resize_y, 3), output_channels=3, filters=64,
                    norm_type='instancenorm', num_blocks=9):
    inputs = layers.Input(shape=input_shape, name='input_image')
    #input_2 = layers.Input(shape=input_shape, name='input_image_2')

    #combined_inputs = tf.keras.layers.concatenate([inputs, input_2])  # (bs, 256, 256, channels*2)

    x = layers.Conv2D(filters * filter_multiplier, 7, strides=1, padding='same')(inputs)
    x = layers.ReLU()(x)

    # Downsampling
    x = layers.Conv2D(filters * 2 * filter_multiplier, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters * 4 * filter_multiplier, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(num_blocks):
        x = ResidualBlock(x, filters * 4 * filter_multiplier, 3, norm_type)

    # Upsampling
    x = layers.Conv2DTranspose(filters * 2 * filter_multiplier, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(filters * filter_multiplier, 3, strides=2, padding='same')(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(output_channels, 7, padding='same', activation='tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


# ========================Resnet=====================================


def discriminator(filter_multiplier, norm_type='batchnorm', target=True):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
      norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
      target: Bool, indicating whether target image is an input or not.

    Returns:
      Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(math.floor(128 * filter_multiplier), 4, norm_type, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(math.floor(256 * filter_multiplier), 4, norm_type)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(math.floor(512 * filter_multiplier), 4, norm_type)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    # conv = tf.keras.layers.Conv2D(math.floor(512 * filter_multiplier), 4, strides=1, kernel_initializer=initializer,use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)
    conv_layer = tf.keras.layers.Conv2D(math.floor(512 * filter_multiplier), 4, strides=1, kernel_initializer=initializer,
                        use_bias=False)
    spectral_conv = SpectralNormalization(conv_layer)
    conv = spectral_conv(zero_pad1)  # (bs, 31, 31, 512)



    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    #last = tf.keras.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
    final_conv_layer = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, use_bias=False)
    spectral_final_conv = SpectralNormalization(final_conv_layer)
    last = spectral_final_conv(zero_pad2)  # (bs, 30, 30, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)


#
# ###################################################
# def discriminator(filter_multiplier):
#     initializer = tf.random_normal_initializer(0., 0.02) #where mean is 0 and the STD is 0.02
#     inp = tf.keras.layers.Input(shape=[256,256,3], name='input_image')
#     tar = tf.keras.layers.Input(shape=[256,256,3], name='target_image')
#     x = tf.keras.layers.concatenate([inp, tar])
#     down1 = downsample(128* filter_multiplier, 4, 'instancenorm')(x) # fordi vi har en batch size på 128,128,64
#     down2 = downsample(256* filter_multiplier, 4)(down1) #batch size 64,64,128
#     down3 = downsample(512* filter_multiplier, 4)(down2) #batch size ,32,32,256
#
#     zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
#     conv = SpectralNormalization(tf.keras.layers.Conv2D(512* filter_multiplier, 4, strides=1, kernel_initializer=initializer, use_bias=False))(zero_pad1) #batch size ,31,31,512
#     norm1 = InstanceNormalization()(conv)
#     leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
#     zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) #batchsize,33,33,512
#     last = SpectralNormalization(tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, use_bias=False))(zero_pad2) #batch size 30,30,1
#     return tf.keras.Model(inputs=[inp,tar], outputs=[last])
#
# ####################################################




# ==================================prøve pix2pix example fra tensorflow authors==================================

if generator_type == 'unet':
    generator_g = unet_generator(filter_muultiplier_generator, OUTPUT_CHANNELS,
                                 norm_type="instancenorm")  # Generator() #pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    generator_f = unet_generator(filter_muultiplier_generator, OUTPUT_CHANNELS,
                                 norm_type="instancenorm")  # Generator()  #pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

if generator_type == 'resnet':
    generator_g = ResNetGenerator(filter_muultiplier_generator)
    generator_f = ResNetGenerator(filter_muultiplier_generator)

discriminator_x = discriminator(filter_muultiplier_discriminator)#,training= False)# norm_type='instancenorm',
                                #target=False)  # pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = discriminator(filter_muultiplier_discriminator)#, training = False)# norm_type='instancenorm',
                                #target=False)  # pix2pix.discriminator(norm_type='instancenorm', target=False)

###################################################################
#                    exponential average generator weights
###################################################################


shadow_generator_g = tf.keras.models.clone_model(generator_g)
shadow_generator_g.set_weights(generator_g.get_weights())
shadow_generator_f = tf.keras.models.clone_model(generator_f)
shadow_generator_f.set_weights(generator_f.get_weights())

def update_shadow_weights(shadow_model, model, beta=0.999):
    shadow_weights = shadow_model.get_weights()
    model_weights = model.get_weights()
    updated_shadow_weights = [beta * shadow_weight + (1 - beta) * model_weight for shadow_weight, model_weight in zip(shadow_weights, model_weights)]
    shadow_model.set_weights(updated_shadow_weights)


# endregion
try:
    print(f"sample_simulated shape: {sample_simulated.shape}")

except:
    print(f"sample_simulated len: {len(sample_simulated)}")
_,input_x = sample_simulated

_,input_y= sample_train   #input_img , real_img

to_training_image = generator_g(input_x)
to_simulated = generator_f(input_y)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_simulated, to_training_image, sample_train, to_simulated]
title = ['sample_simulated', 'To to_training_image', 'sample_train', ' To simulated']

# for i in range(len(imgs)):
#     plt.subplot(2, 2, i + 1)
#     plt.title(title[i])
#     if i % 2 == 0:
#         plt.imshow(imgs[i][0] * 0.5 + 0.5)
#     else:
#         plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()
#
# plt.figure(figsize=(8, 8))
# plt.subplot(121)
# plt.title('Is a real oil_drum?')
# plt.imshow(discriminator_y(sample_train)[0, ..., -1], cmap='RdBu_r')
#
# plt.subplot(122)
# plt.title('Is a fake oildrum?')
# plt.imshow(discriminator_x(sample_simulated)[0, ..., -1], cmap='RdBu_r')
#
# plt.show()



def gradient_penalty(discriminator, real_images, fake_images, lambda_gp=10.0):
    batch_size = real_images.shape[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
    interpolated_images = real_images * alpha + fake_images * (1 - alpha)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        predictions = discriminator(interpolated_images, training=True)
    gradients = tape.gradient(predictions, [interpolated_images])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    #gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2) #centered around a gradientnorm of 1
    gradient_penalty = tf.reduce_mean(tf.square(slopes)) # centered around a gradientnorm of 0 slik som paperet https://arxiv.org/pdf/2303.16280.pdf foreslår


    return gradient_penalty * lambda_gp



# region define loss functions
# loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_obj = tf.keras.losses.MeanSquaredError()


# def discriminator_loss(real, generated):
#     real_loss = loss_obj(tf.ones_like(real), real)
#
#     generated_loss = loss_obj(tf.zeros_like(generated), generated)
#
#     total_disc_loss = real_loss + generated_loss
#
#     return total_disc_loss * 0.5



 # def discriminator_loss(discriminator, real, generated, real_images, fake_images, lambda_gp=10.0): #lambda_gp=10.0 brukes i paperet om gradient penalty https://arxiv.org/pdf/1704.00028.pdf
 #    real_loss = loss_obj(tf.ones_like(real), real)
 #    generated_loss = loss_obj(tf.zeros_like(generated), generated)
 #    gp = gradient_penalty(discriminator, real_images, fake_images, lambda_gp)
 #    total_disc_loss = real_loss + generated_loss + gp
 #    return total_disc_loss * 0.5

# ##########################################

def discriminator_loss(disc_real_output,disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss
##########################################

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


# endregion

generator_g_optimizer = tf.keras.optimizers.Adam(learningrate_G_g, beta_1=beta_G_g)
generator_f_optimizer = tf.keras.optimizers.Adam(learningrate_G_f, beta_1=beta_G_f)

discriminator_x_optimizer = tf.keras.optimizers.Adam(learningrate_D_x, beta_1=beta_D_x)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learningrate_D_y, beta_1=beta_D_y)


# generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
#
# discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
# discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# checkpoint_path = "./checkpoints/train"
#
# ckpt = tf.train.Checkpoint(generator_g=generator_g,
#                            generator_f=generator_f,
#                            discriminator_x=discriminator_x,
#                            discriminator_y=discriminator_y,
#                            generator_g_optimizer=generator_g_optimizer,
#                            generator_f_optimizer=generator_f_optimizer,
#                            discriminator_x_optimizer=discriminator_x_optimizer,
#                            discriminator_y_optimizer=discriminator_y_optimizer)
#
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')

"""
Training
"""



# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    print("Aktiveringsvektorer form:", act1.shape, act2.shape)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    print("Kovariansmatrise form:", sigma1.shape, sigma2.shape)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        print("Komplekse tall funnet i covmean")
        covmean = covmean.real

    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


FIDmodel = InceptionV3(include_top=False, pooling='avg', input_shape=(resize_x, resize_y, 3))

def calculate_fid_single_image(model, image1, image2):
    # calculate feature vectors
    f1 = model.predict(image1)
    f2 = model.predict(image2)

    # calculate the outer product of f1 and f2
    outer_f1 = np.outer(f1, f1)
    outer_f2 = np.outer(f2, f2)


    # calculate sum squared difference between feature vectors
    ssdiff = np.sum((f1 - f2) ** 2.0)

    # calculate the norm of difference between outer products
    norm_diff = np.linalg.norm(outer_f1 - outer_f2, 'fro') # Frobenius norm

    # custom FID score combining the differences
    custom_fid = ssdiff + norm_diff
    return custom_fid


############################################################################################################
#                   calculate SSIM and PSNR for images to measure the quality of the generated images
############################################################################################################

# def calculate_SSIM_and_PSNR(imageA, imageB):
#     ssim_index = ssim(imageA, imageB, data_range=imageB.max() - imageB.min())
#     psnr_value = psnr(imageA, imageB, data_range=imageB.max() - imageB.min())
#
#     print(f"SSIM: {ssim_index}, PSNR: {psnr_value}")
#     return ssim_index, psnr_value


def generate_images(model, input, epoch_num, num, testing=False):
    #################################################
    #                   training
    #################################################
    if testing == False:
        _,test_input = input

        def log_wrapper(name, value, step):
            if isinstance(step, tf.Tensor):
                step = step.numpy()
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            run[name].log(value, step=int(step))

        # assert test_input.shape[1:] == (256, 256, 3), f"Input shape was: {test_input.shape}, expected: (256, 256, 3)"
        prediction = model(test_input)

        #=================
        test_input_prepared = preprocess_input(test_input)
        prediction_prepared = preprocess_input(prediction)
        #================

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']
        num_elem = len(display_list)

        for i in range(num_elem):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
            folder_name = 'generated_data/generated_images_cycle_GAN_simulated_dataset'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if epoch_num % save_every_n_epochs == 0:
            # plt.savefig(os.path.join(folder_name, ' image_at_epoch_{:04d}.png'.format(epoch)))
            print(f'fig saved at epoch {epoch_num}')
            print(f"epoch type: {type(epoch_num)}")
            # plt.close("all")
            # Save the figure using the step number to keep track of progress
            plt.savefig(f'{folder_name}/test image_at_step_{epoch_num:04d}.png')
            image_path_buffer = f'{folder_name}/test image_at_step_{epoch_num:04d}.png'
            run[f"visualizations/from_training/test_image_at_step_{epoch_num:04d}"].upload(image_path_buffer)
            if BATCH_SIZE > 1 or BATCH_SIZE_TEST > 1:
                fid_score = calculate_fid(FIDmodel, test_input_prepared, prediction_prepared)
                print("FID Score using several pics:", fid_score)
                tf.py_function(func=log_wrapper, inp=[f"train/FID_score", fid_score, tf.convert_to_tensor(epoch_num, dtype=tf.int32)], Tout=[])
            else:
                fid_score = calculate_fid_single_image(FIDmodel, test_input_prepared, prediction_prepared)
                print(f"FID Score one to one in training state at epoch {epoch_num}: ", fid_score)
                tf.py_function(func=log_wrapper, inp=[f"train/FID_score", fid_score, tf.convert_to_tensor(epoch_num, dtype=tf.int32)], Tout=[])
        plt.show()

    #################################################
    #                   testing
    #################################################
    if testing:
        # =========================================================================
        # test_input = tf.expand_dims(test_input, axis=0)
        # assert test_input.shape[1:] == (256, 256, 3), f"Input shape was: {test_input.shape}, expected: (256, 256, 3)"
        print("plotting test images")

        def log_wrapper(name, value, step):
            if isinstance(step, tf.Tensor):
                step = step.numpy()
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            run[name].log(value, step=int(step))

        # assert test_input.shape[1:] == (256, 256, 3), f"Input shape was: {test_input.shape}, expected: (256, 256, 3)"
        prediction = model(input)

        # =================
        test_input_prepared = preprocess_input(test_input)
        prediction_prepared = preprocess_input(prediction)
        # ================

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Test Input Image', 'Predicted Image from testinput']
        num_elem = len(display_list)

        for i in range(num_elem):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
            folder_name = 'generated_data/testing/generated_images_cycle_GAN_simulated_dataset'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # plt.savefig(os.path.join(folder_name, ' image_at_epoch_{:04d}.png'.format(epoch)))
            print('fig saved')
            # plt.close("all")
            # Save the figure using the step number to keep track of progress
            plt.savefig(f'{folder_name}/test image_at_step_{num:04d}.png')
            image_path_buffer = f'{folder_name}/test image_at_step_{num:04d}.png'
            run[f"visualizations/test_my_model/test_image_at_step_{num:04d}"].upload(image_path_buffer)
            if BATCH_SIZE_TEST > 1:
                fid_score = calculate_fid(FIDmodel, test_input_prepared, prediction_prepared)
                print("FID Score tested:", fid_score)
                tf.py_function(func=log_wrapper, inp=[f"test/FID_score", fid_score, tf.convert_to_tensor(num, dtype=tf.int32)], Tout=[])
            else:
                fid_score = calculate_fid_single_image(FIDmodel, test_input_prepared, prediction_prepared)
                print("FID Score one to one:", fid_score)
                tf.py_function(func=log_wrapper, inp=[f"test/FID_score", fid_score, tf.convert_to_tensor(num, dtype=tf.int32)], Tout=[])

        # plt.close()  # Close the figure to free up memory
        # print('Saved generated images at step '+ str(step))
        plt.show()

# =============================================================================


@tf.function
def train_step(real_x, real_y, lambda_gp=10):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.


        ###############################
        target_x, input_x = real_x
        print("target_x shape:", target_x.shape, "input_x shape:", input_x.shape)
        target_y, input_y = real_y  # input_img , real_img


        ################################

        fake_y = generator_g(input_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(input_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(input_x, training=True)
        same_y = generator_g(input_y, training=True)

        disc_real_x = discriminator_x(inputs=[input_x, same_x], training=True)
        disc_real_y = discriminator_y(inputs=[input_y, same_y], training=True)

        disc_fake_x = discriminator_x([target_x, fake_x], training=True)
        disc_fake_y = discriminator_y([target_y,fake_y], training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        # disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        # disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
        # disc_x_loss = discriminator_loss(discriminator_x, disc_real_x, disc_fake_x, input_x, fake_x, lambda_gp)
        # disc_y_loss = discriminator_loss(discriminator_y, disc_real_y, disc_fake_y, input_y, fake_y, lambda_gp)

        ###########################################
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss,
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss,
                                              discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                              generator_f.trainable_variables))


    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))




    # ==============logging===========================================================

    def log_wrapper(name, value):
        # Denne funksjonen vil bli kalt av tf.py_function, så den kan inneholde eager-kode.
        run[name].log(value.numpy())

    # Bruker en lambda-funksjon som wrapper for å unngå å passere Run objektet direkte
    for name, value in [("train/gen_g_loss", gen_g_loss),
                        ("train/gen_f_loss", gen_f_loss),
                        ("train/total_cycle_loss", total_cycle_loss),
                        ("train/disc_x_loss", disc_x_loss),
                        ("train/disc_y_loss", disc_y_loss)]:
        tf.py_function(lambda v: log_wrapper(name, v), [value], [])

    return {
        'gen_g_loss': gen_g_loss,
        'gen_f_loss': gen_f_loss,
        'disc_x_loss': disc_x_loss,
        'disc_y_loss': disc_y_loss,
        'total_cycle_loss': total_cycle_loss
    }


for epoch in range(EPOCHS):
    start = time.time()

    total_gen_g_loss = 0
    total_gen_f_loss = 0
    total_disc_x_loss = 0
    total_disc_y_loss = 0
    total_cycle_loss = 0

#######################################################
#                   learning rate decay after 100 epochs på generator og discriminator
#######################################################


    if epoch >= 100:

        new_lr = max(learningrate_D_x * (1 - (epoch - 100) / 100), 0)  # Sørger for at læringstakten ikke går under 0
        new_lr_g = max(learningrate_G_g * (1 - (epoch - 100) / 100), 0)
        tf.keras.backend.set_value(discriminator_x_optimizer.learning_rate, new_lr)
        tf.keras.backend.set_value(discriminator_y_optimizer.learning_rate, new_lr)
        tf.keras.backend.set_value(generator_g_optimizer.learning_rate, new_lr_g)
        tf.keras.backend.set_value(generator_f_optimizer.learning_rate, new_lr_g)

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((simulated_dataset, train_dataset)):
        #train_step(image_x, image_y)
        losses = train_step(image_x, image_y)

        total_gen_g_loss += losses['gen_g_loss']
        total_gen_f_loss += losses['gen_f_loss']
        total_disc_x_loss += losses['disc_x_loss']
        total_disc_y_loss += losses['disc_y_loss']
        total_cycle_loss += losses['total_cycle_loss']
        n += 1
        if n % 10 == 0:
            print('.', end='', flush=True)
        #n += 1

    avg_gen_g_loss = total_gen_g_loss / n
    avg_gen_f_loss = total_gen_f_loss / n
    avg_disc_x_loss = total_disc_x_loss / n
    avg_disc_y_loss = total_disc_y_loss / n
    avg_cycle_loss = total_cycle_loss / n

    run["epoch/avg_gen_g_loss_for_epoch"].log(avg_gen_g_loss)
    run["epoch/avg_gen_f_loss_for_epoch"].log(avg_gen_f_loss)
    run["epoch/avg_disc_x_loss_for_epoch"].log(avg_disc_x_loss)
    run["epoch/avg_disc_y_loss_for_epoch"].log(avg_disc_y_loss)
    run["epoch/avg_cycle_loss_for_epoch"].log(avg_cycle_loss)


    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    num = 0

    ###################################################################
    #                    exponential average generator weights
    ###################################################################
    # if epoch >= 1:
    #     update_shadow_weights(shadow_generator_g, generator_g, beta=0.60)
    #     update_shadow_weights(shadow_generator_f, generator_f, beta=0.60)
    # elif epoch  < 1:
    #     shadow_generator_g = generator_g
    #     shadow_generator_f = generator_f

    update_shadow_weights(shadow_generator_g, generator_g, beta=0.99)
    update_shadow_weights(shadow_generator_f, generator_f, beta=0.99)


    #generate_images(generator_g, sample_simulated, epoch, num)
    generate_images(shadow_generator_g, sample_simulated, epoch, num)


    # if (epoch + 1) % 5 == 0:
    #   ckpt_save_path = ckpt_manager.save()
    #   print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
    #                                                        ckpt_save_path))

    print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                       time.time() - start))

end_time = time.time()  # Lagrer slutttiden
elapsed_time = end_time - start_time  # Beregner tiden det tok å kjøre koden

print(f"Tiden det tok å kjøre koden: {elapsed_time / 60} minutter, eller {(elapsed_time / 60) / 60} timer")
print("training done===============================================")
print("Generate using test dataset")

num = 0
print(f"len test dataset: {len(test_dataset)}")

for test_batch in test_dataset.take(len(test_dataset)): #endre ved behov
    # Siden generate_images forventer et enkelt bilde, pass test_image direkte
    generate_images(generator_g, test_batch, epoch, num, testing=True)

    num += 1

# Lagrer modellene til fil
generator_g.save(f'saved_model_cycle_GAN/{image_type[1:-8]}/my_generator_g.h5')
discriminator_x.save(f'saved_model_vanilla_GAN/{image_type[1:-8]}/my_discriminator_x.h5')
generator_f.save(f'saved_model_cycle_GAN/{image_type[1:-8]}/my_generator_f.h5')
discriminator_y.save(f'saved_model_vanilla_GAN/{image_type[1:-8]}/my_discriminator_y.h5')

run["cycle_GAN/generator_g"].upload(f'saved_model_cycle_GAN/{image_type[1:-8]}/my_generator_g.h5')
run["vanilla_GAN/discriminator_x"].upload(f'saved_model_cycle_GAN/{image_type[1:-8]}/my_discriminator_x.h5')
run["cycle_GAN/generator_f"].upload(f'saved_model_cycle_GAN/{image_type[1:-8]}/my_generator_f.h5')
run["vanilla_GAN/discriminator_y"].upload(f'saved_model_cycle_GAN/{image_type[1:-8]}/my_discriminator_y.h5')




run.stop()

time.sleep(30)  # for at slurm i fox ikke skal avslutte jobben før neptune har gjort seg ferdig
print("======================= End of line ===================================")
