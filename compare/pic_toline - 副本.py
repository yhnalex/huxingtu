#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

from PIL import Image
from PIL import ImageEnhance


#获取尺寸线
def get_axis_line(list,dis,high,piexl, times, scope,num):
    #list:在范围内每个像素点的个数，数组类型 ；dis：分布范围，即x范围；high：y最大限值；piexl：尺寸线范围像素； times：尺寸线与周围像素个数的比值； scope：探寻范围是分布范围的多大一部分；num：尺寸线or尺寸界线
    #尺寸线
    if num == 1:
        list_line = []
        for i in range(len(list)):
            for w in range(int(dis / scope)):
                if i > w - 1 and i < len(list) - w:
                    if (list[i] > times * list[i - w]) and list[i] > high / 20:
                        for w2 in range(piexl):
                            if i + w2 < len(list):
                                if (list[i] > times * list[i + w2]):
                                    list_line.append(i)
                    elif ((list[i] > times * list[i - w]) or (list[i] > times * list[i + w])) and list[i] > high / 20:
                        list_line.append(i)
        return(list_line)
    #尺寸线界限
    elif num == 2:
        list_line = []
        for i in range(len(list)):
            for w in range(piexl):
                if i > w - 1 and i < len(list) - w:
                    if ((list[i] > times * list[i - w]) or (list[i] > times * list[i + w])) and list[i] > high / 20:
                        list_line.append(i)
        return (list_line)

# 获取尺寸线的左右边界线
def get_axis_line_zuoyoulist(list_line,list,des, scope):
    list_line_list = []
    for k in list_line:
        list_line_item = [0,0]
        list_line_item_min = [10000 for i in range(len(list))]
        list_line_item_max = [10000 for i in range(len(list))]
        for w in range(int(des / scope)):
            if k > w - 1:
                if (list[k] >= 10 * list[k - w]):
#                     if list_line_item[0] == 0:
                    list_line_item_min[k - w] = list[k - w]
#                         list_line_item[0] = 1
#                 if list[k - w] > list[k - w + 1]:
#                     list_line_item[0] = 0
            if k < len(list) - w:
                if (list[k] >= 10 * list[k + w]):
#                     if list_line_item[1] == 0:
                    list_line_item_max[k + w] = list[k + w]
#                         list_line_item[1] = 1
#                 if list[k + w] > list[k + w - 1]:
#                     list_line_item[1] = 0
        try:
            list_line_item = [min(get_all_index(list_line_item_min, min(list_line_item_min))), max(get_all_index(list_line_item_max, min(list_line_item_max)))]
        except Exception as e:
            print (e)
        list_line_list.append(list_line_item)


    #去重
    zuoyoulist = []
    for i in list_line_list:
        if i not in zuoyoulist and 0 < i[0] < des and 0 < i[1] < des:
            zuoyoulist.append(i)
    zuoyoulist.sort()
    #融合
    zuoyoulist = get_list_fix(zuoyoulist)
    return zuoyoulist

def get_min_without_zero(list_all):
    a = []
    for x in list_all:
        if x > 0:
            a.append(x)
    b = min(a)
    return b

def get_all_index(list_all, target):
    a = [i for i, x in enumerate(list_all) if x == target]
    return a


#([23,53],[38,64] 变成 [23,64])
def get_list_fix(zuoyoulist):
    zuoyoulist = list(zuoyoulist)
    zuoyoulist_fix = []
    location = 0
    for i in range(len(zuoyoulist)):
        for k in range(len(zuoyoulist)-i):
            if i + k + 1 < len(zuoyoulist) and i == location:
                if zuoyoulist[i][1] >= zuoyoulist[i+k+1][0]:
                    zuoyoulist[i][1] = zuoyoulist[i+k+1][1]
                elif zuoyoulist[i][1] < zuoyoulist[i+k+1][0]:
                    zuoyoulist_fix_item = zuoyoulist[i]
                    zuoyoulist_fix.append(zuoyoulist_fix_item)
                    location = i+k+1
            elif i + k + 1 == len(zuoyoulist) and i == location:
                zuoyoulist_fix_item = zuoyoulist[i]
                zuoyoulist_fix.append(zuoyoulist_fix_item)
    return zuoyoulist_fix

  # 获取方向标尺上下界线_all
def get_axis_line_shagnxialist(zuoyoulist,dis,piexl,thresh,num):
    #zuoyoulist：左右边界线;dis:分布范围;piexl:线条宽度;thresh:cv2.threshold;num:1:适用于寻找垂直方向标尺上下界线，2：适用于寻找水平方向界线
    if num == 1:
        # 灰度水平方向投影
        x_list = []
        y_list = []
        for k in zuoyoulist:
            lenth = k[1]-k[0]
            zuoyoulist_shangxia_item = [0 for z in range(0, dis)]
            for j in range(0, dis):
                for i in range(0, lenth):
                    if thresh[j, k[0]+i] == 0:
                        zuoyoulist_shangxia_item[j] += 1
            x_k = range(dis)
            y_k = zuoyoulist_shangxia_item
            x_list.append(x_k)
            y_list.append(y_k)

        shagnxia_list = []
        # 获取垂直方向标尺上下界线
        for k in range(len(zuoyoulist)):
            line_shagnxia_list_item = get_axis_line(y_list[k], dis, zuoyoulist[k][1]-zuoyoulist[k][0], piexl, 10, 10, 2)
            line_shagnxia_list_item_min = min(line_shagnxia_list_item)
            line_shagnxia_list_item_max = max(line_shagnxia_list_item)
            shagnxia_list.append((line_shagnxia_list_item_min, line_shagnxia_list_item_max))
        return shagnxia_list
    if num == 2:
        # 灰度垂直方向投影
        x_list = []
        y_list = []
        for k in zuoyoulist:
            lenth = k[1]-k[0]
            zuoyoulist_shangxia_item = [0 for z in range(0, dis)]
            for j in range(0, dis):
                for i in range(0, lenth):
                    if thresh[k[0]+i, j] == 0:
                        zuoyoulist_shangxia_item[j] += 1
            x_k = range(dis)
            y_k = zuoyoulist_shangxia_item
            x_list.append(x_k)
            y_list.append(y_k)

        shagnxia_list = []
        # 获取垂直方向标尺上下界线
        for k in range(len(zuoyoulist)):
            line_shagnxia_list_item = get_axis_line(y_list[k], dis, zuoyoulist[k][1]-zuoyoulist[k][0], piexl, 10, 10, 2)
            line_shagnxia_list_item_min = min(line_shagnxia_list_item)
            line_shagnxia_list_item_max = max(line_shagnxia_list_item)
            shagnxia_list.append((line_shagnxia_list_item_min, line_shagnxia_list_item_max))
        return shagnxia_list



##去除尺寸标尺
def get_without_rule(img, threshold_value,piexl):
    ret, thresh1 = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    (height, weight) = thresh1.shape  # 返回高和宽

    # 灰度垂直方向投影
    ver = [0 for z in range(0, weight)]
    # ver = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数
    # 记录每一列的波峰
    for j in range(0, weight):  # 遍历一列
        for i in range(0, height):  # 遍历一行
            if thresh1[i, j] == 0:  # 如果改点为黑点
                ver[j] += 1  # 该列的计数器加一计数+

    # 灰度水平方向投影
    lev = [0 for z in range(0, height)]
    for j in range(0, height):
        for i in range(0, weight):
            if thresh1[j, i] == 0:
                lev[j] += 1

    # 获取垂直尺寸线
    line_ver = get_axis_line(ver, weight, height, piexl, 30, 20, 1)
    print(line_ver)
    # 获取垂直尺寸线左右界线
    ver_zuoyoulist = get_axis_line_zuoyoulist(line_ver, ver, weight, 20)
    print(ver_zuoyoulist)
    # 获取垂直尺寸线上下界线
    ver_shagnxia_list = get_axis_line_shagnxialist(ver_zuoyoulist, height, piexl, thresh1, 1)


    x1 = range(len(ver))
    y1 = ver

    x2 = range(len(lev))
    y2 = lev

    x_list = []
    x_list.append(x1)
    x_list.append(x2)

    y_list = []
    y_list.append(y1)
    y_list.append(y2)





    # 获取水平尺寸线
    line_lev = get_axis_line(lev, height, weight, piexl, 30, 20, 1)
    # 获取水平尺寸线上下界线
    lev_zuoyoulist = get_axis_line_zuoyoulist(line_lev, lev, height, 20)
    # 获取水平尺寸线左右界线
    lev_shangxia_list = get_axis_line_shagnxialist(lev_zuoyoulist, weight, piexl, thresh1, 2)


    contours = []
    for k in range(len(ver_zuoyoulist)):
        contours_item = [(ver_zuoyoulist[k][0], ver_shagnxia_list[k][0]), (ver_zuoyoulist[k][0], ver_shagnxia_list[k][1]), (ver_zuoyoulist[k][1],ver_shagnxia_list[k][1]), (ver_zuoyoulist[k][1], ver_shagnxia_list[k][0])]
        contours.append(contours_item)

    for k in range(len(lev_zuoyoulist)):
        contours_item = [(lev_shangxia_list[k][0], lev_zuoyoulist[k][0]), (lev_shangxia_list[k][0], lev_zuoyoulist[k][1]), (lev_shangxia_list[k][1],lev_zuoyoulist[k][1]), (lev_shangxia_list[k][1], lev_zuoyoulist[k][0])]
        contours.append(contours_item)

    fig = plt.figure(figsize=(10, 20))
    for j in range(2):
        ax = fig.add_subplot(2,1,j+1)
        ax.plot(x_list[j], y_list[j], c='r')
        ax.set_title('第' + str(j))
    plt.show()

    print('img', img)

    # for j in range(0, weight):  # 遍历一列
    #     for i in range(0, height):  # 遍历一行
    #         if j in var_white or lev[i] < 10:
    #             img.itemset((i, j), 255)

    pts = np.array(contours, np.int32)
    img = cv2.polylines(img, pts, True, (0, 255, 255))
    cv2.imshow('image', img)
    cv2.waitKey(0)

img=cv2.imread('D:/huxingku_train/34851a69041bbc08b0055d67455af1df.jpg',cv2.IMREAD_GRAYSCALE)
get_without_rule(img,220,5)
