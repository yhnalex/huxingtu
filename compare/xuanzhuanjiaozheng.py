# 先通过hough transform检测图片中的图片，计算直线的倾斜角度并实现对图片的旋转
import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage
from matplotlib import pyplot as plt


def get_rotate_img(fliepath):
    img = cv2.imread(fliepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 霍夫变换

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        if x1 == x2 or y1 == y2:
            return img
        t = float(y2 - y1) / (x2 - x1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        rotate_img = ndimage.rotate(img, rotate_angle)
        return rotate_img

def suofang(img):
    size = img.shape
    height = size[0]
    weight = size[1]
    times  = int(max(height, weight)/300)
    pic = cv2.resize(img, (int(height/times), int(weight/times)), interpolation=cv2.INTER_CUBIC)
    return pic

def quzao(img):
    dst = cv2.fastNlMeansDenoising(img, 10)
    return dst

def mohu(img):
    blur = cv2.bilateralFilter(img, 10, 75, 120)
    return blur

def ruihua(img):
    # laplacian = cv2.Laplacian(img, cv2.CV_64F)  # CV_64F为图像深度
    #
    # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # 1，0参数表示在x方向求一阶导数
    #
    # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # 0,1参数表示在y方向求一阶导数

    # plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # return  laplacian

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst = cv2.filter2D(img, -1, kernel=kernel)
    return  dst