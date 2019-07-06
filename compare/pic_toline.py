#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

from PIL import Image
from PIL import ImageEnhance


def getMatchNum(matches,ratio,points1f,points2f):
    #返回特征点匹配数量和匹配掩码'''
    points1f_list=[[0,0] for i in range(len(matches))]
    points2f_list=[[0,0] for i in range(len(matches))]
    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    a1=a2=b1=b2=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            matchesMask[i]=[1,0]
            matchNum+=1
            points1f_list[i] = points1f[m.queryIdx]
            points2f_list[i] = points2f[n.trainIdx]
            a1 = a1 + points1f_list[i][0]
            b1 = b1 + points1f_list[i][1]
            a2 = a2 + points1f_list[i][0]
            b2 = b2 + points1f_list[i][1]
    points1f_main = (a1 / matchNum, b1 / matchNum)
    points2f_main = (a2 / matchNum, b2 / matchNum)
    return (matchNum,matchesMask,points1f_main,points2f_main,points1f_list,points2f_list)

def get_easy_img(img):
    white_img = np.zeros(img.copy().shape, np.uint8)
    ret, binary = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 20:
            cv2.drawContours(white_img, [contours[i]], -1, (255, 0, 0), 3)
    print(contours)
    return white_img

def get_position_score(matches,ratio,points1f_main,points2f_main,points1f_list,points2f_list):
    p1_main = np.array(points1f_main)
    p2_main = np.array(points2f_main)
    p1 = np.array(points1f_list)
    p2 = np.array(points2f_list)
    points1f_list_tran = np.array([[0, 0] for i in range(len(matches))])
    score = np.zeros(len(matches))
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            p1_way = p1[i] - p1_main
            p1_len = math.hypot(p1_way[0], p1_way[1])
            p2_way = p2[i] - p2_main
            p2_len = math.hypot(p2_way[0], p2_way[1])
            points1f_list_tran[i] = p2_main + p1_way*(p2_len/p1_len)
            diff = p2[i] - points1f_list_tran[i]
            difflen = math.hypot(diff[0], diff[1])
            if p1_len+p2_len > 0:
                score[i] = 1 - (difflen ** 2/(p1_len **2 + p2_len ** 2))**(1/2)
            else:
                pass
    score_non_zero = (score>0)
    score = np.mean(score_non_zero)*100
    return score

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
        print('list_line:',list_line)
        return(list_line)
    #尺寸线界限
    elif num == 2:
        list_line = []
        for i in range(len(list)):
            for w in range(piexl):
                if i > w - 1 and i < len(list) - w:
                    if ((list[i] > times * list[i - w]) or (list[i] > times * list[i + w])) and list[i] > high / 20:
                        list_line.append(i)
        print('list_line:', list_line)
        return (list_line)

# 获取尺寸线的左右边界线
def get_axis_line_zuoyoulist(list_line,list,des, scope):
    list_line_list = []
    for k in list_line:
        list_line_item = [0, 0]
        list_line_item_min = [0 for i in range(len(list))]
        list_line_item_max = [0 for i in range(len(list))]
        for w in range(int(des / scope)):
            if k > w - 1 and k < len(list) - w:
                if (list[k] >= 10 * list[k - w]):
                    if list_line_item[0] == 0:
                        list_line_item_min[k - w] = list[k - w]
                    elif list_line_item[0] != 0 and list[k - w] > list[k - w + 1]:
                        list_line_item[0] = 0
                if (list[k] >= 10 * list[k + w]):
                    if list_line_item[1] == 0:
                        list_line_item_max[k + w] = list[k + w]
                    elif list_line_item[1] != 0 and list[k + w] > list[k + w - 1]:
                        list_line_item[1] = 0
        print(list_line_item_min)
        print(list_line_item_max)
        list_line_item = min(get_all_index(list_line_item_min, get_min_without_zero(list_line_item_min))), max(get_all_index(list_line_item_max, get_min_without_zero(list_line_item_max)))
        list_line_list.append(list_line_item)
    print('list_line_list',list_line_list)
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
    if len(a) = 0:
        return
    b = min(a)
    return b

def get_all_index(list_all, target):
    a = [i for i, x in enumerate(list_all) if x == target]
    return a

#([23,53],[38,64] 变成 [23,64])
def get_list_fix(zuoyoulist):
    zuoyoulist = zuoyoulist
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
        print(y_list)
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
        print(y_list)
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
    line_ver = get_axis_line(ver, weight, height, piexl, 50, 40, 1)
    print('line_ver:', line_ver)
    # 获取垂直尺寸线左右界线
    ver_zuoyoulist = get_axis_line_zuoyoulist(line_ver, ver, weight)
    print('ver_zuoyoulist:', ver_zuoyoulist)
    # 获取垂直尺寸线上下界线
    ver_shagnxia_list = get_axis_line_shagnxialist(ver_zuoyoulist, height, piexl, thresh1, 1)
    print('ver_shagnxia_list:', ver_shagnxia_list)


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
    line_lev = get_axis_line(lev, height, weight, piexl, 30, 40, 1)
    print('line_lev:', line_lev)
    # 获取水平尺寸线上下界线
    lev_zuoyoulist = get_axis_line_zuoyoulist(line_lev, lev, height)
    print('lev_zuoyoulist:', lev_zuoyoulist)
    # 获取水平尺寸线左右界线
    lev_shangxia_list = get_axis_line_shagnxialist(lev_zuoyoulist, weight, piexl, thresh1, 2)
    print('lev_shangxia_list:', lev_shangxia_list)


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

path='D:/huxingku_train/sift/'
queryPath=path+'' #图库路径
samplePath=path+'5a00305a-aa0a-4bdc-aa28-f1fbb0db79f7.jpg' #样本图片
comparisonImageList=[] #记录比较结果



#创建SIFT特征提取器
sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000, nOctaveLayers=3, contrastThreshold=0.1, edgeThreshold=8, sigma=1.6)
#创建FLANN匹配对象
FLANN_INDEX_KDTREE=0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams,searchParams)

sampleImage = cv2.imread(samplePath, cv2.IMREAD_GRAYSCALE)
#white_img = get_easy_img(sampleImage)
white_img = sampleImage
kp1, des1 = sift.detectAndCompute(white_img, None) #提取样本图片的特征
points1f = cv2.KeyPoint_convert(kp1)
picsort = []
for parent,dirnames,filenames in os.walk(queryPath):
    i = 0
    for p in filenames:
        i = i+1
        try:
            p=queryPath+p
            queryImage=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
            #white_img2 = get_easy_img(queryImage)
            white_img2 = queryImage
            kp2, des2 = sift.detectAndCompute(white_img2, None) #提取比对图片的特征
            points2f = cv2.KeyPoint_convert(kp2)
            matches=flann.knnMatch(des1,des2,k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
            (matchNum, matchesMask, points1f_main, points2f_main, points1f_list, points2f_list)=getMatchNum(matches,0.8,points1f,points2f) #通过比率条件，计算出匹配程度和中心点以及匹配坐标
            loc_score = get_position_score(matches,0.8,points1f_main,points2f_main,points1f_list,points2f_list)
            matchRatio=matchNum*100/len(matches)
            drawParams=dict(matchColor=(0,255,0),
                    singlePointColor=(255,0,0),
                    matchesMask=matchesMask,
                    flags=0)
            final_score = loc_score*0.6+matchRatio*0.4
            comparisonImage=cv2.drawMatchesKnn(white_img,kp1,white_img2,kp2,matches,None,**drawParams)
            comparisonImageList.append((p,comparisonImage,final_score)) #记录下结果
            print(final_score,loc_score,matchRatio)
            picsort.append((p, final_score))

        except Exception as e:
            print(Exception, e)
        print(p,'   ',i, '/', len(filenames))

picsort.sort(key=lambda x:x[1],reverse=True)
print(picsort[0][1])

comparisonImageList.sort(key=lambda x:x[2],reverse=True) #按照匹配度排序
count=len(comparisonImageList)
print(comparisonImageList[0][0])
column=4
row=math.ceil(count/column)
#绘图显示
figure,ax=plt.subplots(row,column)
for index,(name,image,ratio) in enumerate(comparisonImageList):
    ax[int(index/column)][index%column].set_title('Similiarity %.2f%%' % ratio)
    ax[int(index/column)][index%column].imshow(image)
    print(name)
plt.show()

