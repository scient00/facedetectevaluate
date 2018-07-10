#!/usr/bin/env python
# -*- coding: utf-8 -*-
from detection.FaceDetectInput import ConfigType, vDetectInfo, Rect, DetectInfo
import yaml,pandas,logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# 读取配置文件(测试类别、阈值)
def readConfigFile(confpath = 'config/config.yaml'):
    try:
        with open(confpath,'r') as cofile:
            conf = yaml.load(cofile)
        return ConfigType(conf.get('CLASSES_TO_EVAL'),conf.get('IOU_LEVELS'))
    except:
        return ConfigType()

# 读取检测/标注数据
def readTestData(detepath,typeName = None):
    try:
        vDeteList = []
        with open(detepath,'r') as detefile:
            allData = detefile.readlines()
        for line in allData:
            imageName,objNum,rectlist = line.split()[0],line.split()[1],line.split()[2:]
            vDetect = vDetectInfo(imageName)
            deteinfolist = [rectlist[i:i+5] for i in range(0,len(rectlist),5)]
            for deteinfo in deteinfolist:
                x = float(deteinfo[0])
                y = float(deteinfo[1])
                width= float(deteinfo[2])
                height = float(deteinfo[3])
                confience = float(deteinfo[4])
                deteOut = DetectInfo(typeName, Rect(x, y, width, height), confience)
                vDetect.add(deteOut)
            vDeteList.append(vDetect)

        return vDeteList
    except:
        logging.error('readTestData error!!!')
        return vDetectInfo()
