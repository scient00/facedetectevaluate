#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import *

class Rect:
    '''
     矩形框坐标信息
    '''
    def __init__(self, x=0.0, y=0.0, width=0.0, height=0.0):
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height

    def __repr__(self):
        return 'x=' + str(self.__x) + ' y=' + str(self.__y) + ' width=' + str(self.__width) + ' height=' + str(
            self.__height)

    def set(self, x, y, width, height):
        self.__x = x
        self.__y = y
        self.__width = width
        self.__height = height

    def getX(self):
        return self.__x

    def getY(self):
        return self.__y

    def getWidth(self):
        return self.__width

    def getHeight(self):
        return self.__height

    def toTuple(self):
        return self.__x, self.__y, self.__width, self.__height

    def toIntArr(self):
        oRect = (c_int * 4)()
        oRect[0] = self.__x
        oRect[1] = self.__y
        oRect[2] = self.__width
        oRect[3] = self.__height
        return oRect

class Point:
    '''
     坐标点信息
    '''
    def __init__(self, x=0, y=0):
        self.__x = x
        self.__y = y

    def __repr__(self):
        return 'x=' + str(self.__x) + ' y=' + str(self.__y)

    def set(self, x, y):
        self.__x = x
        self.__y = y

    def getX(self):
        return self.__x

    def getY(self):
        return self.__y

    def toTuple(self):
        return self.__x, self.__y

class DetectInfo(object):
    '''
     检测信息
    '''
    def __init__(self, typeName=None, rect=Rect(), confience=0.0):
        self.__typeName = typeName  # 检测类型
        self.__rect = rect  # 检测位置
        self.__confience = confience  # 置信度

    def __repr__(self):
        return 'typeName:' + str(self.__typeName) + '\trect:' + str(self.__rect) + '\tConfience:' + str(
            self.__confience)

    def set(self, typeName, rect):
        self.__typeName = typeName
        self.__rect = rect

    def getTypeName(self):
        return self.__typeName

    def getRect(self):
        return self.__rect

    def getConfence(self):
        return self.__confience

    def toTuple(self):
        return self.__typeName, self.__rect, self.__confience

class vDetectInfo(object):
    def __init__(self, imageName=None):
        self.__imageName = imageName
        self.__size = 0
        self.__vOut = []

    def __repr__(self):
        return 'imageName:' + str(self.__imageName) + ' size:' + str(self.__size) + '\t' + str(self.__vOut)

    def add(self, detectinfo=DetectInfo()):
        self.__size += 1
        self.__vOut.append(detectinfo.toTuple())

    def getImageName(self):
        return self.__imageName

    def getSize(self):
        return self.__size

    def getDetectInfo(self, index):
        return DetectInfo(self.__vOut[index][0], self.__vOut[index][1], self.__vOut[index][2])

class ConfigType(object):
    '''
     配置文件读取
    '''
    def __init__(self, classes=None, thresholdValue=None, imageRoi=None, detectScale=None):
        self.__classes = classes  # 测试类别
        self.__thresholdValue = thresholdValue  # 阈值

    def __repr__(self):
        return str(self.__classes) + '\n' + str(self.__thresholdValue) + '\n'

    def getClasses(self):
        return self.__classes

    def getIouValue(self):
        return self.__thresholdValue
