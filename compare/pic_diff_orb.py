#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import numpy as np

import cv2
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
    print(matchNum)
    print('------')
    print(matchesMask)
    return matchNum, matchesMask

path='D:/huxingku_train/pic/'
queryPath=path+'test/' #图库路径
samplePath=path+'20180708142707424701531032164842.jpg' #样本图片
comparisonImageList=[] #记录比较结果

#创建ORB特征提取器
orb = cv2.ORB_create()
#创建FLANN匹配对象
FLANN_INDEX_LSH=6
indexParams=dict(algorithm=FLANN_INDEX_LSH,
                 table_number = 6, #12
                 key_size = 12,    #20
                 multi_probe_level = 1)#2
searchParams=dict(checks=100)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)

sampleImage=cv2.imread(samplePath,0)
kp1, des1 = orb.detectAndCompute(sampleImage, None) #提取样本图片的特征
for parent,dirnames,filenames in os.walk(queryPath):
    for p in filenames:
        try:
            p=queryPath+p
            queryImage=cv2.imread(p,0)
            kp2, des2 = orb.detectAndCompute(queryImage, None) #提取比对图片的特征
            matches=flann.knnMatch(des1, des2, k=2) #匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
            (matchNum, matchesMask)= getMatchNum(matches, 0.8) #通过比率条件，计算出匹配程度
            matchRatio=matchNum*100/len(matches)
            drawParams=dict(matchColor=(0,255,0),
                    singlePointColor=(255,0,0),
                    matchesMask=matchesMask,
                    flags=0)
            comparisonImage=cv2.drawMatchesKnn(sampleImage,kp1,queryImage,kp2,matches,None,**drawParams)
            comparisonImageList.append((comparisonImage,matchRatio)) #记录下结果
        except Exception as e:
            print(Exception, e)

comparisonImageList.sort(key=lambda x:x[1],reverse=True) #按照匹配度排序
count=len(comparisonImageList)
column=4
row=math.ceil(count/column)
#绘图显示
figure,ax=plt.subplots(row,column)
for index,(image,ratio) in enumerate(comparisonImageList):
    ax[int(index/column)][index%column].set_title('Similiarity %.2f%%' % ratio)
    ax[int(index/column)][index%column].imshow(image)
plt.show()
