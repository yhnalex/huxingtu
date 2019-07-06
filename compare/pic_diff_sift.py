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

