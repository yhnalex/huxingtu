import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt

def hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    print(hist)
    plt.subplot(121)
    plt.imshow(img,'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title("Original")
    plt.subplot(122)
    plt.plot(hist)
    plt.show()
    return hist

def mean_normaliztion(list):
    arr = np.array(list)
    list_mean_normaliztion = []
    for x in arr:
        x = float(x - np.min(arr))/(np.max(arr)- np.min(arr))
        list_mean_normaliztion.append(x)
    return list_mean_normaliztion

def gen(signal,window_size):
    index = ans = times = 0
    while True:
        while index <window_size:
            ans += signal[times + index]
            index += 1
        yield float(ans)/float(window_size)
        ####Reset
        index = 0
        ans = 0
        times +=1

def mean_filter(signal,window_size):
    temp = gen(signal, window_size)
    filtered = []
    for i in range(len(signal)-window_size):
        filtered.append(next(temp))
    return filtered

def find_peak(signal,lenth,thre,peak_width):
    l = []
    l_list = []
    print (signal)
    for i in range(0, lenth-1):
        if i == 0:
            l.append(i)
        else:
            if signal[i-1] < signal[i] and signal[i] > signal[i+1] and signal[i] > thre:
                l.append(i) #找出极值
            elif signal[i] == signal [i-1] and signal[i] > thre:
                l.append(i)
    print('l:', l)
    for j in range(0, len(l)-1):
        if l[j+1] - l[j] >= peak_width:
            l_list.append(l[j])
    l_list.append(l[len(l)-1])
    l_list = list(set(l_list))
    return l_list

#获取波谷
def get_low(signal,lenth,thre,peak_width):
    l = []
    l_list = []
    print(signal)
    for i in range(0, lenth - 1):
        if i == 0:
            l.append(i)
        else:
            if signal[i - 1] > signal[i] and signal[i] < signal[i + 1] and signal[i] < thre:
                l.append(i)  # 找出极值
            elif signal[i] == signal[i - 1] and signal[i] < thre:
                l.append(i)
    print('l:', l)
    for j in range(0, len(l) - 1):
        if l[j + 1] - l[j] >= peak_width:
            l_list.append(l[j])
    l_list.append(l[len(l) - 1])
    l_list.sort()
    print('l_list:', l_list)
    return l_list

def get_peak(signal, window_size):
    signal = mean_normaliztion(signal)
    filter_signal = mean_filter(signal, window_size)
    plt.plot(signal)
    plt.plot(filter_signal)
    plt.show()
    lenth = len(filter_signal)
    thre = 0.1
    peak_width = 20
    peak_list = get_low(filter_signal, lenth, thre, peak_width)
    print('peak_list:', peak_list)
    return peak_list

def get_real_peak(signal, peak_list, window_size, peak_width):
    peak_list_num = []
    signal = signal.tolist()
    peak_list_num_real = []
    for i in range(len(peak_list)):
        peak_list_item = []
        for w in range(window_size):
            peak_list_item.append(signal[peak_list[i] - w])
            peak_list_item.append(signal[peak_list[i] + w])
        peak_list_item_sig = min(peak_list_item)
        peak_list_num.append(signal.index(peak_list_item_sig))
        peak_list_num.append(0)
        for j in range(0, len(peak_list_num) - 1):
            if peak_list_num[j + 1] - peak_list_num[j] >= peak_width:
                peak_list_num_real.append(peak_list_num[j])
        peak_list_num_real.append(peak_list_num[len(peak_list_num) - 1])

        peak_list_num_real = list(set(peak_list_num_real))
        peak_list_num_real.sort()
    print('peak_list_num_real:', peak_list_num_real)
    return peak_list_num_real

def get_hist_qiang(img, peak_list_num, signal):
    signal = signal.tolist()
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    height = img.shape[0]
    width = img.shape[1]
    count_list = []
    for k in range(len(peak_list_num)-1):
        count = check_isnot_qiangti(img, height, width, peak_list_num[k], peak_list_num[k+1], thresh1)
        count_list.append(count)
    print('peak_list_num:', peak_list_num)
    print('count_list:', count_list)
    cv2.imwrite('D:/huxingku_train/test2_.jpg', thresh1)

    qinagti_hist_min = peak_list_num[count_list.index(max(count_list))]
    qinagti_hist_max = peak_list_num[count_list.index(max(count_list))+1]
    print([qinagti_hist_min, qinagti_hist_max])
    qiangtimin = 0
    qiangtimax = 0
    # for k in range(255):
    #     if abs(max_count_list - check_isnot_qiangti(img, height, width, qinagti_hist - k, thresh1)) > 10:
    #         qiangtimin = qinagti_hist - k
    #         break
    # for k in range(255):
    #     if abs(max_count_list - check_isnot_qiangti(img, height, width, qinagti_hist + k, thresh1)) > 10:
    #         qiangtimax = qinagti_hist + k
    #         break
    print(qinagti_hist_min,qinagti_hist_max)
    for i in range(1, height):  # 遍历每一行
        for j in range(1, width):  # 遍历每一列
            if qinagti_hist_min <= img[i][j] <= qinagti_hist_max:
                img[i][j] = 0
            else:
                img[i][j] = 255

    return img

def check_isnot_qiangti(img,height,width, hist_min, hist_max, thresh):
    count = 0
    for i in range(1, height-1):  # 遍历每一行
        for j in range(1, width-1):  # 遍历每一列
            if hist_min <= img[i][j] <= hist_max:
                if isnot_qianti(thresh, i, j):
                    count += 1
    return count


def isnot_qianti(thresh, i, j):
    if ((thresh[i-1][j-1] ==255 and thresh[i-1][j] ==255 and thresh[i-1][j+1] ==255 and thresh[i][j-1] ==255 and thresh[i][j] == 0 and thresh[i][j+1] == 0 and thresh[i+1][j-1] == 255 and thresh[i+1][j] == 0 and thresh[i+1][j+1] ==0)
    or (thresh[i-1][j-1] ==255 and thresh[i-1][j] ==255 and thresh[i-1][j+1] ==255 and thresh[i][j-1] ==0 and thresh[i][j] == 0 and thresh[i][j+1] == 255 and thresh[i+1][j-1] == 0 and thresh[i+1][j] == 0 and thresh[i+1][j+1] ==255)
    or (thresh[i-1][j-1] ==0 and thresh[i-1][j] ==0 and thresh[i-1][j+1] ==255 and thresh[i][j-1] ==0 and thresh[i][j] == 0 and thresh[i][j+1] == 255 and thresh[i+1][j-1] == 255 and thresh[i+1][j] == 255 and thresh[i+1][j+1] ==255)
    or (thresh[i-1][j-1] ==255 and thresh[i-1][j] ==0 and thresh[i-1][j+1] ==0 and thresh[i][j-1] ==255 and thresh[i][j] == 0 and thresh[i][j+1] == 0 and thresh[i+1][j-1] == 255 and thresh[i+1][j] == 255 and thresh[i+1][j+1] ==255)
    or (thresh[i-1][j-1] ==0 and thresh[i-1][j] ==0 and thresh[i-1][j+1] ==0 and thresh[i][j-1] ==0 and thresh[i][j] == 0 and thresh[i][j+1] == 0 and thresh[i+1][j-1] == 255 and thresh[i+1][j] == 255 and thresh[i+1][j+1] ==255)
    or (thresh[i-1][j-1] ==255 and thresh[i-1][j] ==0 and thresh[i-1][j+1] ==0 and thresh[i][j-1] ==255 and thresh[i][j] == 0 and thresh[i][j+1] == 0 and thresh[i+1][j-1] == 255 and thresh[i+1][j] == 0 and thresh[i+1][j+1] ==0)
    or (thresh[i-1][j-1] ==255 and thresh[i-1][j] ==255 and thresh[i-1][j+1] ==255 and thresh[i][j-1] ==0 and thresh[i][j] == 0 and thresh[i][j+1] == 0 and thresh[i+1][j-1] == 0 and thresh[i+1][j] == 0 and thresh[i+1][j+1] ==0)
    or (thresh[i-1][j-1] ==0 and thresh[i-1][j] ==0 and thresh[i-1][j+1] ==255 and thresh[i][j-1] ==0 and thresh[i][j] == 0 and thresh[i][j+1] == 255 and thresh[i+1][j-1] == 0 and thresh[i+1][j] == 0 and thresh[i+1][j+1] ==255)):
        return True
    else:
        return False

def qiangtishibie(img):
    signal = hist(img)[0:127]
    peak_list_num = get_real_peak(signal, get_peak(signal, 8), 8, 20)
    print(peak_list_num)
    img2 = get_hist_qiang(img, peak_list_num, signal)
    return img2
    # cv2.imshow('img2.jpg', img2)
    # cv2.waitKey(0)


