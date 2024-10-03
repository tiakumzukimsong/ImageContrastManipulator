# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib inline
from IPython.display import Image

img_bgr = cv2.imread("new_zealand_coast.jpg",cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

Image(filename="new_zealand_coast.jpg")

matrix = np.ones(img_rgb.shape, dtype = "int8") * 50
matrix = matrix.astype(img_rgb.dtype)
img_rgb_brighter = cv2.add(img_rgb,matrix)
img_rgb_darker = cv2.subtract(img_rgb,matrix)

plt.figure(figsize=[18,8])
plt.subplot(131);plt.imshow(img_rgb_darker); plt.title("Darker")
plt.subplot(132);plt.imshow(img_rgb); plt.title("original")
plt.subplot(133);plt.imshow(img_rgb_brighter); plt.title("Brighter")

matrix1 = np.ones(img_rgb.shape) * .8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb),matrix1))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb),matrix2))

plt.figure(figsize=[18,8])
plt.subplot(131);plt.imshow(img_rgb_darker); plt.title("lower contrast")
plt.subplot(132);plt.imshow(img_rgb); plt.title("original")
plt.subplot(133);plt.imshow(img_rgb_brighter); plt.title("higher contrast")
