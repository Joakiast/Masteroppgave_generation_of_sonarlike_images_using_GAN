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



from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import random

#from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import peak_signal_noise_ratio as psnr


# Set seeds for reproducibility
seed_number = 42

tf.random.set_seed(seed_number)
np.random.seed(seed_number)
random.seed(seed_number)

run = neptune.init_run(
    project="masteroppgave/testRun",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMDY3ZDFlNS1hMGVhLTQ1N2YtODg4MC1hNThiOTM1NGM3YTQifQ=="
)
#
# # def upload_plot_to_neptune(imgs, titles, contrast=8):
# #     plt.figure(figsize=(8, 8))
# #     for i in range(len(imgs)):
# #         plt.subplot(2, 2, i+1)
# #         plt.title(titles[i])
# #         if i % 2 == 0:
# #             plt.imshow(imgs[i][0] * 0.5 + 0.5)
# #         else:
# #             plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# #         plt.axis('off')
# #
# #     # Lagre til en BytesIO stream og laste opp direkte til Neptune
# #     img_buffer = BytesIO()
# #     plt.savefig(img_buffer, format='png')
# #     img_buffer.seek(0)
# #     run["visualizations/generator_outputs"].upload(img_buffer)
# #     plt.close()
#
#
# start_time = time.time()
# # region load the dataset
# # test 0
# resize_x = 256
# resize_y = 256
#
# # The bath size of 1 gives better results using the UNet in this experiment.
# BATCH_SIZE = 1
# BATCH_SIZE_TEST = BATCH_SIZE
# EPOCHS = 200
# decay_start_epoch = 100
# color_channel = 3
# crop_size = 256  # resize_x / 2 150 fin størrelse på
# DROPOUT = 0#.5
# LAMBDA = 10
#
# learningrate_G_g = 0.0002  # 7e-5
# learningrate_G_f = 0.0002  # 7e-5
# learningrate_D_x = learningrate_G_g / 2  # 4e-5
# learningrate_D_y = learningrate_G_f / 2  # 4e-5
#
# beta_G_g = 0.9
# beta_G_f = 0.9
# beta_D_x = 0.9
# beta_D_y = 0.9
#
# save_every_n_epochs = 2
#
# generator_type = "resnet" # virker som resnet gir best resultat
# #generator_type = "unet"
#
# filter_muultiplier_generator = 2
# filter_muultiplier_discriminator = 1
#
# # image_type = '*rock_RGB'
# image_type = '*oil_drum_RGB'
# # image_type = '*clutter_RGB'
# # image_type = "*man_made_object_RGB"
# # image_type = "*Mine_size_rock_RGB.jpg"
#
# #image_type_2 = False
# # image_type_2 = '*rock_RGB'
# # image_type_2 = '*oil_drum_RGB'
# # image_type_2 = "*man_made_object_RGB"
# image_type_2 = "*Mine_size_rock_RGB.jpg"
#
#
# image_type_3 = False
# # image_type_3 = '*rock_RGB'
# # image_type_3 = '*oil_drum_RGB'
# # image_type_3 = "*man_made_object_RGB"
# # image_type = "*Mine_size_rock_RGB.jpg"
#
#
# params = {
#     "activation": "tanh",
#     "n_epochs": EPOCHS,
#     "decay_start_epoch": decay_start_epoch,
#     "batch_size": BATCH_SIZE,
#     "drop_out": DROPOUT,
#     "learningrate_generator_g": learningrate_G_g,
#     "learningrate_generator_f": learningrate_G_f,
#     "learningrate_discriminator_x": learningrate_D_x,
#     "learningrate_discriminator_y": learningrate_D_y,
#     "beta_G_g": beta_G_g,
#     "beta_G_f": beta_G_f,
#     "beta_D_x": beta_D_x,
#     "beta_D_y": beta_D_y,
#     "Lambda": LAMBDA,
#     "Image_type added with extra set": image_type,
#     "use_bias": False,
#     #  "number_of_filters": "increased x2 in generator not discriminator",
#     "type of generator": generator_type,
#     "type of loss func": "MeanSquaredError",
#     "save_every_n_epochs": save_every_n_epochs,
#     "filter multiplier gen": filter_muultiplier_generator,
#     "filter multiplier disc": filter_muultiplier_discriminator,
#     "seed_number": seed_number
# }
#
# if image_type_2:
#     params["image_type_2"] = image_type_2
#
# if image_type_3:
#     params["image_type_3"] = image_type_3
#
# run["model/parameters"] = params
#
# # region Preparing datasets
#
# train_set_path = pathlib.Path("datasets/train")
# train_set_path_simulated = pathlib.Path("datasets/barrel_sim_v2/sim_data_rgb_barrel_v2")#("datasets/sim_data_rgb_barrel") kommentert ut gammel simulert datasett
# train_set_path_simulated_v1 = pathlib.Path("datasets/sim_data_rgb_barrel")
# test_set_path_simulated = pathlib.Path("datasets/test_set_cycleGAN")
# #test_set_path = pathlib.Path("datasets/test")
# #test_set_path_handdrawn = pathlib.Path("datasets/image_translation_handdrawn_images")
# train_set_extra_path = pathlib.Path("datasets/test")
#
# image_paths_train = [str(path) for path in list(train_set_path.glob(image_type + ".jpg"))]  # [:8000]  # filterer ut data i datasettet i terminal: ls |grep oil
# print(f"size of trainingset: {len(image_paths_train)}")
#
# image_paths_train_extra = [str(path) for path in list(
#     train_set_extra_path.glob(image_type + ".jpg"))]  # [:8000]  # filterer ut data i datasettet i terminal: ls |grep oil
#
# image_paths_train_sim_V1 = [str(path) for path in list(
#     train_set_path_simulated_v1.glob("*.png"))]
# print(f"size of simulert trainingset V1: {len(image_paths_train_sim_V1)}")
#
# image_paths_train.extend(image_paths_train_extra)
# print(f"size of sonar trainingset after adding extra training data: {len(image_paths_train)}")
#
#
# if image_type_2:
#     img_buffer_1 = [str(path) for path in list(train_set_path.glob(image_type_2 + ".jpg"))]  # [:8000]
#     image_paths_train.extend(img_buffer_1)
# if image_type_3:
#     img_buffer_2 = [str(path) for path in list(train_set_path.glob(image_type_3 + ".jpg"))]  # [:8000]
#     image_paths_train.extend(img_buffer_2)
#
# image_paths_train_simulated = [str(path) for path in list(train_set_path_simulated.glob("*.png"))]  # total størrelse 425   # filterer ut data i datasettet i terminal: ls |grep oil
# print(f"size of simulated trainingset:: {len(image_paths_train_simulated)}")
# image_paths_train_simulated.extend(image_paths_train_sim_V1)
# print(f"size of simulated trainingset after adding extra simulated data: {len(image_paths_train_simulated)}")
#
# image_paths_test =[str(path) for path in list(test_set_path_simulated.glob("*.png"))] #[str(path) for path in list(train_set_path_simulated.glob("*.png"))][553:]  # filterer ut data i datasettet i terminal: ls |grep oil
# print(f"size of testset: {len(image_paths_test)}")
#
# #buffer_test = [str(path) for path in list(test_set_path_handdrawn.glob(
#  #   "*.png"))]  # [:405] #total størrelse 425   # filterer ut data i datasettet i terminal: ls |grep oil
# #image_paths_test.extend(buffer_test)
#
#
# def crop_image_around_POI(image, point_x, point_y, crop_size):
#     # Konverter punktkoordinater til heltall
#     point_x = tf.cast(point_x, tf.int32)
#     point_y = tf.cast(point_y, tf.int32)
#     crop_size = tf.cast(crop_size, tf.int32)
#
#     # Beregn øvre venstre hjørne av beskjæringsboksen
#     start_y = tf.maximum(0, point_y - crop_size // 2)
#     start_x = tf.maximum(0, point_x - crop_size // 2)
#
#     # Sørg for at beskjæringsboksen ikke går utenfor bildet
#     image_height, image_width, _ = image.shape
#     image_height = tf.cast(image_height, tf.int32)
#     image_width = tf.cast(image_width, tf.int32)
#
#     if start_x + crop_size > image_width:
#         start_x = tf.maximum(0, image_width - crop_size)
#     if start_y + crop_size > image_height:
#         start_y = tf.maximum(0, image_height - crop_size)
#
#     image = tf.image.crop_to_bounding_box(image, start_y, start_x, crop_size, crop_size)
#
#     # print(f"crop by {crop_size}")
#     return image
#
#
# def find_coordinates_for_cropping_tensor(path_image):
#     # base_name = str[path_image]#tf.cast(path_image, str)
#     base_name_b = os.path.basename(path_image.numpy())
#     base_name = base_name_b.decode("utf-8")
#     # print(f"base name {base_name}")
#     label_file = base_name.replace('.jpg', '.txt')  # Bytt ut filendelsen fra .jpg til .txt
#     # print(f"label file {label_file}")
#
#     if hasattr(path_image, 'numpy'):
#         path_image_str = path_image.numpy().decode("utf-8")
#     else:
#         path_image_str = path_image
#
#     #####################
#     if "test" in path_image_str:
#         label_path = os.path.join("datasets/test/Label", label_file)
#     else:
#         label_path = os.path.join("datasets/train/Label", label_file)
#     #####################
#
#
#     #label_path = os.path.join("datasets/train/Label", label_file)
#     # print(f"label_path {label_path}")
#     x, y = None, None
#     try:
#
#         with open(label_path, 'r') as file:
#             label_content = file.read()
#
#         for line in label_content.split('\n'):
#             parts = line.split()
#             # if parts and parts[0] != 'clutter':
#             x, y = map(float, parts[1:3])
#             # print(f"x: {x}, y: {y}")
#             return x, y
#
#     except Exception as e:
#         print(f"Error while processing label file {label_path}: {e}")
#     return None, None
#
#
# def find_coordinates_for_cropping(path_image):
#     base_name = os.path.basename(path_image)  # .numpy())
#     # print(f"base name {base_name}")
#     label_file = base_name.replace('.jpg', '.txt')  # Bytt ut filendelsen fra .jpg til .txt
#     # print(f"label file {label_file}")
#
#     if hasattr(path_image, 'numpy'):
#         path_image_str = path_image.numpy().decode("utf-8")
#     else:
#         path_image_str = path_image
#
#     ####################
#     if "test" in path_image_str:
#         label_path = os.path.join("datasets/test/Label", label_file)
#     else:
#         label_path = os.path.join("datasets/train/Label", label_file)
#     #####################
#
#
#     #label_path = os.path.join("datasets/train/Label", label_file)
#     # print(f"label_path {label_path}")
#     x, y = None, None
#     try:
#
#         with open(label_path, 'r') as file:
#             label_content = file.read()
#
#         for line in label_content.split('\n'):
#             parts = line.split()
#             # if parts and parts[0] != 'clutter':
#             x, y = map(float, parts[1:3])
#             # print(f"x: {x}, y: {y}")
#             return x, y
#             # elif parts and parts[0] == 'rock':
#             #     x, y = map(float, parts[1:3])
#             #     return x,y
#
#     except Exception as e:
#         print(f"Error while processing label file {label_path}: {e}")
#     return None, None
#
#
# # ==========================
# def load_and_preprocess_image_trainset(path_image_trainset):
#     # path_simulated_image_trainset = pathlib.Path("datasets")
#     if isinstance(path_image_trainset, tf.Tensor):
#         # print("===================start load and preprocess image============================================")
#         image = tf.io.read_file(path_image_trainset)
#         image = tf.image.decode_jpeg(image,
#                                      channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
#
#         image = tf.cast(image, tf.float32)
#         image = (image - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
#         # if not "clutter" in image_type:
#         # x,y = tf.py_function(func=find_coordinates_for_circle_tensor, inp=[path_image_trainset], Tout=[tf.float32, tf.float32])
#         x, y = tf.py_function(func=find_coordinates_for_cropping_tensor, inp=[path_image_trainset],
#                               Tout=[tf.float32, tf.float32])
#         image.set_shape([400, 600, 3])
#         image = crop_image_around_POI(image, x, y, crop_size)
#         # inp_image = tf.py_function(func = remove_part_of_image, inp = [image,x,y], Tout = tf.float32)#remove_part_of_image(image,x,y)#crop_image_around_POI(image, x, y, crop_size)
#         # inp_image.set_shape([400, 600, 3])
#         image = tf.image.resize(image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
#         # ====================for inp===============================================================================
#
#         return image
#     else:
#         # print("===================start load and preprocess image============================================")
#         image = tf.io.read_file(path_image_trainset)
#         image = tf.image.decode_jpeg(image,
#                                      channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
#         image = tf.cast(image, tf.float32)
#         image = (image - 127.5) / 127.5  # Normaliser bildene til [-1, 1] området
#         assert image.shape == (400, 600, 3)
#         # image = tf.image.resize(image, [400, 600], method=tf.image.ResizeMethod.AREA)
#         # if not "clutter" in image_type:
#         x, y = find_coordinates_for_cropping(path_image_trainset)
#         image = crop_image_around_POI(image, x, y, crop_size)
#         # image = tf.image.resize(image, [resize_x,resize_y], method=tf.image.ResizeMethod.AREA)
#         # print(f"alle bilder kommer hit: image shape før resize: {image.shape} bilde: {path_image}")
#         image = tf.image.resize(image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
#
#         return image
#
#
# def load_and_preprocess_image_simulated_set(path_simulated_image_trainset):
#     # path_simulated_image_trainset = pathlib.Path("datasets")
#     if isinstance(path_simulated_image_trainset, tf.Tensor):
#         # print("===================start load and preprocess image============================================")
#
#         inp_image = tf.io.read_file(path_simulated_image_trainset)
#         inp_image = tf.image.decode_png(inp_image,
#                                         channels=color_channel)  # Bruk tf.image.decode_png for PNG-bilder, etc. endre channels til 3 dersom jeg har rbg bilde
#
#         inp_image = tf.cast(inp_image, tf.float32)
#         inp_image = (inp_image - 127.5) / 127.5
#         # if not "clutter" in image_type:
#         inp_image.set_shape([369, 496, 3])
#         inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
#         return inp_image
#     else:
#         # print("===================start load and preprocess image============================================")
#
#         # =================================for inp image======================================================
#         inp_image = tf.io.read_file(path_simulated_image_trainset)
#         inp_image = tf.image.decode_png(inp_image, channels=color_channel)
#         # print(f" inp_image.shape: {inp_image.shape}")
#         inp_image = tf.cast(inp_image, tf.float32)
#         inp_image = (inp_image - 127.5) / 127.5
#         assert inp_image.shape == (369, 496, 3)
#         inp_image = tf.image.resize(inp_image, [resize_x, resize_y], method=tf.image.ResizeMethod.LANCZOS5)
#
#         return inp_image
#
#
# BUFFER_SIZE_trainset = len(image_paths_train)
# print(f"BUFFER_SIZE train set:: {BUFFER_SIZE_trainset}")
#
# BUFFER_SIZE_simulated = len(image_paths_train_simulated)
# print(f"BUFFER_SIZE simulated set: {BUFFER_SIZE_simulated}")
#
# BUFFER_SIZE_test_set = len(image_paths_test)
#
#
# def augmentation(input_img):
#     flipped_left_right = tf.image.flip_left_right(input_img)
#     flipped_up_down = tf.image.flip_up_down(input_img)
#     n_degrees = 10  # degrees rotated
#     radians = n_degrees * math.pi / 180
#     rotate = tfa.image.rotate(input_img, -radians)
#
#     # rotate = tf.image.rot90(input_img)
#     # flytte objektet til forksjellige posisjoner i bildet, være forskintg rotering av sonar bilder med tanke på skygge 10 grader opp og ned
#     # kan også være interresant å se på ville augemnteringer som farge og store rotasjoner
#     # kan dele det inn i fysiske realiserbare og ikke realiserbare
#     # kan også være innteresant å croppe og resize så det ikke blir helt likt det blir da ikke fysisk realiserbart, men kan være av interesse
#     # bør ta det til sist liten tro på
#
#     return flipped_left_right, flipped_up_down, rotate
#
#
# inp_augmented_training_data_flip_left_right = []
# real_augmented_training_data_flip_left_right = []
#
# inp_augmented_training_data_flip_up_down = []
# real_augmented_training_data_flip_up_down = []
#
# inp_augmented_training_data_rotate = []
# real_augmented_training_data_rotate = []
#
# i = 0
# for image_path, image_path_simulated in zip(image_paths_train, image_paths_train_simulated):
#     re = load_and_preprocess_image_trainset(image_path)
#     inp = load_and_preprocess_image_simulated_set(image_path_simulated)
#
#     #
#     # # Konverterer begge tensorer til uint8 ved å bruke tf.cast etter å ha skalert dem
#     # inp_uint8 = tf.cast(inp * 255, tf.uint8)
#     # re_uint8 = tf.cast(re * 255, tf.uint8)
#     #
#     # # Konverterer tensorer til numpy-arrays for å kunne vises med matplotlib
#     # inp_numpy = inp_uint8.numpy()
#     # re_numpy = re_uint8.numpy()
#     #
#     # plt.figure(figsize=(10, 5))
#     #
#     # plt.subplot(1, 2, 1)
#     # plt.title("inp")
#     # plt.imshow(inp_numpy)
#     #
#     # plt.subplot(1, 2, 2)
#     # plt.title("re")
#     # plt.imshow(re_numpy)
#     #
#     # plt.show()
#     #
#     # i += 1
#     # if i > 9:
#     #     sys.exit()
#     # tf.py_function(func=find_coordinates_for_cropping_tensor, inp=[path_image], Tout=[tf.float32,tf.float32])
#     # plt.show()
#     # only augment the image if we dont have a image type of clutter
#     if "rock_RGB" in image_type or "oil_drum_RGB" in image_type or "man_made_object_RGB" in image_type:
#         train_flipped_left_right, train_flipped_up_down, train_rotate = augmentation(re)
#         flipped_left_right_train = train_flipped_left_right
#         real_augmented_training_data_flip_left_right.append(flipped_left_right_train)  # , flipped_up_down, rotate])
#         real_augmented_up_down = train_flipped_up_down
#         real_augmented_training_data_flip_up_down.append(real_augmented_up_down)
#         rotate_real = train_rotate
#         real_augmented_training_data_rotate.append(rotate_real)
#
#     elif "clutter" in image_type:
#         pass
#
#     # ===================================for simulated data===================================
#     inp_flipped_left_right, inp_flipped_up_down, inp_rotate = augmentation(
#         inp)  # tf.py_function(func = augmentation, inp = [inp, re], Tout=[tf.float32,tf.float32,tf.float32])
#     flipped_left_right_inp = inp_flipped_left_right
#     inp_augmented_training_data_flip_left_right.append(flipped_left_right_inp)  # , flipped_up_down, rotate])
#     flipped_up_down_inp = inp_flipped_up_down
#     inp_augmented_training_data_flip_up_down.append(flipped_up_down_inp)
#     rotate_inp = inp_rotate
#     inp_augmented_training_data_rotate.append(rotate_inp)
#
#     # ===================================for simulated data===================================
#
# # for image_path_test in image_paths_test:
# #     re_test,inp_test = load_and_preprocess_image(image_path_test)
#
#
# # ======================
# # Opprett et tf.data.Dataset fra bildestier
# # the dataset consist of both inp and re images.
# # region oppretter dataset for trening og inp
# train_dataset = tf.data.Dataset.from_tensor_slices(image_paths_train)
# print(image_paths_train[0])
# print(f"dataset train shape 1: {len(train_dataset)}")
# simulated_dataset = tf.data.Dataset.from_tensor_slices(image_paths_train_simulated)
# print(f"dataset simulated shape 1: {len(simulated_dataset)}")
#
# train_dataset = train_dataset.map(load_and_preprocess_image_trainset, num_parallel_calls=tf.data.AUTOTUNE)
# print(f"dataset shape 2: {len(train_dataset)}")
# simulated_dataset = simulated_dataset.map(load_and_preprocess_image_simulated_set, num_parallel_calls=tf.data.AUTOTUNE)
# print(f"dataset simulated shape 2: {len(simulated_dataset)}")
#
# augmented_training_dataset_flip_left_right = tf.data.Dataset.from_tensor_slices(
#     real_augmented_training_data_flip_left_right)
# augmented_simulated_dataset_flip_left_right = tf.data.Dataset.from_tensor_slices(
#     inp_augmented_training_data_flip_left_right)
# train_dataset = train_dataset.concatenate(augmented_training_dataset_flip_left_right)
# simulated_dataset = simulated_dataset.concatenate(augmented_simulated_dataset_flip_left_right)
# print(f"dataset shape 2: {len(train_dataset)}")
# print(f"dataset simulated shape 2: {len(simulated_dataset)}")
#
# augmented_training_dataset_flip_up_down = tf.data.Dataset.from_tensor_slices(real_augmented_training_data_flip_up_down)
# augmented_simulated_dataset_flip_up_down = tf.data.Dataset.from_tensor_slices(inp_augmented_training_data_flip_up_down)
# train_dataset = train_dataset.concatenate(augmented_training_dataset_flip_up_down)
# simulated_dataset = simulated_dataset.concatenate(augmented_simulated_dataset_flip_up_down)
# print(f"dataset shape 3: {len(train_dataset)}")
# print(f"dataset simulated shape 3: {len(simulated_dataset)}")
#
# augmented_training_dataset_rotate = tf.data.Dataset.from_tensor_slices(real_augmented_training_data_rotate)
# augmented_simulated_dataset_rotate = tf.data.Dataset.from_tensor_slices(inp_augmented_training_data_rotate)
# train_dataset = train_dataset.concatenate(augmented_training_dataset_rotate)
# simulated_dataset = simulated_dataset.concatenate(augmented_simulated_dataset_rotate)
# print(f"dataset shape 4: {len(train_dataset)}")
# print(f"dataset simulated shape 4: {len(simulated_dataset)}")
#
# train_dataset = train_dataset.shuffle(BUFFER_SIZE_trainset)
# simulated_dataset = simulated_dataset.shuffle(BUFFER_SIZE_simulated)
# print(f"dataset shape 5: {len(train_dataset)}")
# print(f"dataset simulated shape 5: {len(simulated_dataset)}")
#
# train_dataset = train_dataset.batch(BATCH_SIZE)
# simulated_dataset = simulated_dataset.batch(BATCH_SIZE)
#
# print(f"dataset shape 6: {len(train_dataset)}")
# print(f"dataset simulated shape 6: {len(simulated_dataset)}")
# train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
# simulated_dataset = simulated_dataset.prefetch(tf.data.AUTOTUNE)
#
# test_dataset = tf.data.Dataset.from_tensor_slices(image_paths_test)
# test_dataset = test_dataset.map(load_and_preprocess_image_simulated_set, num_parallel_calls=tf.data.AUTOTUNE)
# test_dataset = test_dataset.shuffle(BUFFER_SIZE_test_set)
# test_dataset = test_dataset.batch(BATCH_SIZE_TEST)
# test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
#
# # ============================================================================
# # endregion
# number_of_samples_to_show = BATCH_SIZE  # Antall eksempler du ønsker å vise
#
# for i in train_dataset.take(1):
#     print(f"Element type tuple len: {len(i)}")
#
# # # Tar en batch fra datasettet
# # for real_imgs in train_dataset.take(2):
# #     plt.figure(figsize=(10, 5))
# #     for i in range(number_of_samples_to_show):
# #         # Plotter input_img
# #         ax = plt.subplot(2, number_of_samples_to_show, 2 * i + 1)
# #         plt.title("Input Image")
# #         plt.imshow(real_imgs[i].numpy())
# # plt.tight_layout()
# # plt.show()
#
# # for inp_imgs in simulated_dataset.take(2):
# #     plt.figure(figsize=(10, 5))
# #     for i in range(number_of_samples_to_show):
# #         # Plotter input_img
# #         ax = plt.subplot(2, number_of_samples_to_show, 2 * i + 1)
# #         plt.title("Input Image")
# #         plt.imshow(inp_imgs[i].numpy())
# # plt.tight_layout()
# # plt.show()
# # save_path = "/media/joakim/Random Stuff/random_slett_meg"
# #
# # # Sjekk om mappen eksisterer, hvis ikke, opprett den
# # if not os.path.exists(save_path):
# #     os.makedirs(save_path)
# #
# # # Anta at last_500_dataset er ditt TensorFlow Dataset
# # count = 0
# # for img in train_dataset:
# #     # Opprett et filnavn for hvert bilde
# #     filename = os.path.join(save_path, f"image_{count}.png")
# #
# #     # Lagre bildet
# #     plt.figure()
# #     plt.imshow(img.numpy().squeeze())  # Juster dette kallet basert på ditt datasett
# #     plt.axis('off')
# #     plt.savefig(filename)
# #     plt.close()  # Lukk figuren for å frigjøre minne
# #
# #     # Inkrementer tellevariabelen
# #     count += 1
# #
# #     # (Valgfritt) Gi tilbakemelding i konsollen
# #     if count % 100 == 0:
# #         print(f"Lagret {count} bilder...")
#
# sample_simulated = next(iter(simulated_dataset))
# sample_train = next(iter(train_dataset))
# sample_test = next(iter(test_dataset))
#
#
# # endregion
#
#
# """
# Import and reuse the Pix2Pix models
# Import the generator and the discriminator used in Pix2Pix via the installed tensorflow_examples package.
#
# The model architecture used in this tutorial is very similar to what was used in pix2pix. Some of the differences are:
#
# Cyclegan uses instance normalization instead of batch normalization.
# The CycleGAN paper uses a modified resnet based generator. This tutorial is using a modified unet generator for simplicity.
# """
# OUTPUT_CHANNELS = 3

import matplotlib.pyplot as plt
import numpy as np

# Generate a random image
random_image = np.random.rand(10, 10)

# Plot the random image
plt.imshow(random_image, cmap='viridis')
plt.colorbar()
plt.show()

run['random_image'].upload('random_image.png')

run.stop()

