#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

from PIL import Image
from PIL import ImageEnhance


import urllib
import ssl
from aip import AipNlp
import pandas as pd
import pymongo
import re
import base64
import json
from urllib import parse

import requests
import json
import base64


""" 你的 APPID AK SK """
APP_ID = '15707569'
API_KEY = 'dM1apxokVTgGGoKwQ08j7Ejr'
SECRET_KEY = 'yNbDlbGM6Qcb50tGlMrlW4Me9GV7DYjp'


# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=dM1apxokVTgGGoKwQ08j7Ejr&client_secret=yNbDlbGM6Qcb50tGlMrlW4Me9GV7DYjp'
request = urllib.request.Request(host)
request.add_header('Content-Type', 'application/json; charset=UTF-8')
response = urllib.request.urlopen(request)
content = response.read()
content_dict = json.loads(content)
if (content):
    print(content)

access_token = content_dict['access_token']
print(access_token)

url_app  = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/detection/get_outside"
request_url = url_app + "?access_token=" + access_token

image = open(r'D:/huxingku_train/20190307204927.jpg', 'rb').read()
data = {'image': base64.b64encode(image).decode()}

#easyDL的API要求post请求的数据格式必须是json字符串数据，因此，在正式请求之前，必须用json.dumps()方法将data_app这个字典转化为json字符串

parms = json.dumps(data)
request = urllib.request.Request(url=request_url, data=parms)
request.add_header('Content-Type', 'application/json')
response = urllib.request.urlopen(request)
content = response.read()
content_dict = json.loads(content)
if (content):
    print(content)

shape = []
score = 0
for location in content_dict['result']:
    if location['name'] == 'huxingtu' and location['score'] > score:
        shape = [(location['height'], location['left'], location['top'], location['width'])]

print(shape)