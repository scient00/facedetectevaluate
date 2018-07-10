#!/usr/bin/env python
# -*- coding: utf-8 -*-

from detection.RunDeteTest import *
import sys,random

if len(sys.argv) <= 1:
    imgpath = 'TestSample/image/'
    labelpath = 'TestSample/labelResult.ly'
    detepath = 'TestSample/deteResult.ly'
else:
    print('参数1:测试图片路径\n参数2:标注样本文件路径\n参数3:检测样本文件路径')
    print('注意：\n\t标注与检测样本存储格式如下:\n\t'
          '\t1.每行存储一条结果，以空格分割\n\t'
          '\t2.存储顺序 [imageName]{[number][x][y][width][height][confidence]}{...}\n\t'
          '\t3.参考示例TestSample/***.ly\n')

    imgpath = sys.argv[1]
    labelpath = sys.argv[2]
    detepath = sys.argv[3]

if __name__ == '__main__':
    detectTest(imgpath,labelpath,detepath)
