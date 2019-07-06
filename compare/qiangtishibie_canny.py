import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

# canny 边缘识别
def qiangtishibie(img):
    white_img = cv2.Canny(img, 100, 300)
    return white_img


img = qiangtishibie(misc.imread('D:/huxingku_train/beautify_img/2.png'))
misc.imsave('D:/huxingku_train/beautify_img/2_canny.png', img)
