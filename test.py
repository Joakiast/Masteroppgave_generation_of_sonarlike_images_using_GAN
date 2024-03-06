



import tensorflow as tf
from keras.src.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

tf.__version__
import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
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

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from PIL import Image
import cv2
from skimage.util import img_as_float


def calculate_SSIM_and_PSNR(imageA, imageB):
    # Forsikre at bildene har samme form.
    if imageA.shape != imageB.shape:
        raise ValueError("Bildene må ha samme dimensjoner for SSIM og PSNR beregninger.")

    # Bestem win_size basert på bildets dimensjoner. Velg en rimelig størrelse gitt bildets størrelse.
    min_dim = min(imageA.shape[:2])  # Ta hensyn til kun de romlige dimensjonene
    win_size = min(11, min_dim - (min_dim % 2 - 1))  # Sørger for at win_size er et oddetall

    # Spesifiser 'channel_axis' for fargebilder. For NumPy arrays, er dette vanligvis -1.
    ssim_index = ssim(imageA, imageB, data_range=imageB.max() - imageB.min(), win_size=win_size, channel_axis=-1)
    psnr_value = psnr(imageA, imageB, data_range=imageB.max() - imageB.min())

    return ssim_index, psnr_value


def compute_structure_component(image1, image2):
    # Konverter bilder til flyttallformat
    image1 = img_as_float(image1)
    image2 = img_as_float(image2)

    # Beregn standardavviket til hvert bilde
    sigmaX = np.std(image1, ddof=1)
    sigmaY = np.std(image2, ddof=1)

    # Beregn samvariasjonen mellom de to bildene
    # np.cov returnerer covariansmatrisen, hvor [0, 1] og [1, 0] er covariansen mellom bildene
    sigmaXY = np.cov(image1.flatten(), image2.flatten(), ddof=1)[0, 1]

    # Beregn strukturkomponenten av SSIM
    structure = sigmaXY / (sigmaX * sigmaY)

    return structure


def load_and_prepare_image(path, resize_dims=(256, 256)):
    # Les inn bildet med matplotlib
    img = plt.imread(path)

    # Hvis bildet er i flyttallformat, normaliser og konverter til uint8
    if img.dtype == np.float32:
        # Skaler verdier til 0-255 og konverter til uint8
        img = (255 * img).astype(np.uint8)

    # Hvis bildet har en alfa kanal, fjern den
    if img.shape[-1] == 4:
        img = img[..., :3]  # Fjern alfa kanalen

    # Konverter numpy array til PIL Image
    img_PIL = Image.fromarray(img)

    # Endre størrelsen på bildet
    img_resized = img_PIL.resize(resize_dims, Image.LANCZOS)

    return img_resized


# Bruk den definerte funksjonen for å laste inn og forberede bildene
img_A_path = 'function_testing/image.png'
img_B_path = 'function_testing/img2.png'
imageA = load_and_prepare_image(img_A_path)
imageB = load_and_prepare_image(img_B_path)

# Vis bildene
plt.imshow(imageA)
plt.show()
plt.imshow(imageB)
plt.show()

# Konverter tilbake til numpy arrays for SSIM og PSNR beregninger
imageA_np = np.array(imageA)
imageB_np = np.array(imageB)

# Kall funksjonen og skriv ut resultatene
score1, score2 = calculate_SSIM_and_PSNR(imageA_np, imageB_np)
print(f"SSIM: {score1}, PSNR: {score2}")

score_1 = compute_structure_component(imageA_np, imageB_np)
print(f"Structure Component: {score_1}")

imageA_np = cv2.imread(img_A_path)
imageB_np = cv2.imread(img_B_path)

imageA_np = cv2.GaussianBlur(imageA_np, (5, 5), 0)
imageB_np = cv2.GaussianBlur(imageB_np, (5, 5), 0)

plt.imshow(imageA_np)
plt.show()
plt.imshow(imageB_np)
plt.show()

img_A_canny = cv2.Canny(imageA_np, 120, 200)
img_B_canny = cv2.Canny(imageB_np, 100, 200)

plt.imshow(img_A_canny)
plt.show()
plt.imshow(img_B_canny)
plt.show()