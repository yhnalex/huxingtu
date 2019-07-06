#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np

from matplotlib import pyplot as plt
import os
import math


def getMatchNum(matches,ratio):
    #返回特征点匹配数量和匹配掩码'''
    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance: #将距离比率小于ratio的匹配点删选出来
            matchesMask[i]=[1,0]
            matchNum+=1
    return (matchNum,matchesMask)

def get_canny(img):
    white_img = cv2.Canny(img, 200, 300)
    return white_img

def get_easy_img(img):
    #contours
    white_img = np.ones(img.copy().shape, np.uint8)
    img = cv2.GaussianBlur(img,(5,5),7)
    ret, binary = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
    #binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    # plt.imshow(binary,'gray')
    # plt.show()
    plt.imshow(binary, 'gray')
    plt.show()
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 3000:
            cv2.drawContours(white_img, [contours[i]], -1, (255, 0, 0), 3)
    plt.imshow(white_img,'gray')
    plt.show()
    # print(contours)
    # cv2.imshow('result.jpg',white_img)
    return white_img

path='D:/huxingku_train/pic/'
queryPath=path+'test2/' #图库路径
samplePath=path+'6e8324cf8fa61fa2c69d8844905cb3fc.jpg' #样本图片
comparisonImageList=[] #记录比较结果

#创建SURF特征提取器
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=3000, extended=True)
#创建FLANN匹配对象
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(checks=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)

sampleImage = cv2.imread(samplePath, cv2.IMREAD_GRAYSCALE)
#white_img = get_easy_img(sampleImage)
white_img = get_canny(sampleImage)
#white_img = sampleImage
kp1, des1 = surf.detectAndCompute(white_img, None) #提取样本图片的特征
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
            #white_img2 = queryImage
            white_img2 = get_canny(queryImage)
            kp2, des2 = surf.detectAndCompute(white_img2, None) #提取比对图片的特征
            points2f = cv2.KeyPoint_convert(kp2)
            matches=flann.knnMatch(des1,des2,k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
            (matchNum, matchesMask)=getMatchNum(matches,0.8) #通过比率条件，计算出匹配程度和中心点以及匹配坐标
            matchRatio=matchNum*100/len(matches)
            drawParams=dict(matchColor=(0,255,0),
                    singlePointColor=(255,0,0),
                    matchesMask=matchesMask,
                    flags=0)
            comparisonImage=cv2.drawMatchesKnn(white_img,kp1,white_img2,kp2,matches,None,**drawParams)
            comparisonImageList.append((p,comparisonImage,matchRatio)) #记录下结果

            picsort.append((p, matchRatio))

        except Exception as e:
            print(Exception, e)
        print(i, '/', len(filenames))

picsort.sort(key=lambda x:x[1],reverse=True)
print(picsort[0])

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

