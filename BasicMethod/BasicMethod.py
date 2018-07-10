#!/usr/bin/python
# -*- coding: utf-8 -*-

import os


# 遍历目录下图片
def BrowsePic(fpath):
    PicList = []
    if not os.path.exists(fpath):
        print("此目录不存在!!!")
        return PicList
    for root, dirs, files in os.walk(fpath):
        for x in files:
            if x.find('.jpg') != -1 or x.find('.JPG') != -1 or x.find('.bmp') != -1 or x.find(
                    '.BMP') != -1 or x.find('.png') != -1 or x.find("PNG") != -1:
                PicList.append(root + os.path.sep + x)
        return PicList


# 遍历目录下所有图片，包括子目录
def BrowsePics(fpath):
    PicList = []
    if not os.path.exists(fpath):
        print("此目录不存在!!!")
        return PicList
    for root, dirs, files in os.walk(fpath):
        for x in files:
            if x.find('.jpg') != -1 or x.find('.JPG') != -1 or x.find('.bmp') != -1 or x.find(
                    '.BMP') != -1 or x.find('.png') != -1 or x.find("PNG") != -1:
                PicList.append(root + os.path.sep + x)
    return PicList

# 遍历目录,只遍历一层目录
def BrowseDir(fpath,index = 1):
    DirList = []
    nCount = 0
    if not os.path.exists(fpath):
        print("此目录不存在!!!")
        return DirList
    for root, dirs, files in os.walk(fpath):
        nCount += 1
        if len(dirs) > 0:
            for dir in dirs:
                DirList.append(root + os.path.sep + dir)
        if nCount == index:
            break
    return DirList

# 创建目录
def CreateFolder(path, filename = ' '):
    TempPath = path + os.path.sep + filename
    if not os.path.exists(TempPath):
        # os.mkdir(pathTemp)
        os.makedirs(TempPath)
        return TempPath
    else:
        return TempPath
        print("目录已存在")


# 遍历txt
def BrowseTxt(fpath):
    TxtList = []
    if not os.path.exists(fpath):
        print("此目录不存在!!!")
        return TxtList
    for x in os.listdir(fpath):
        if x.find('.txt') != -1:
            if os.path.isfile(os.path.join(fpath, x)):
                TxtList.append(x)
    return TxtList


# 遍历Xml
def BrowseXml(fpath):
    XmlList = []
    if not os.path.exists(fpath):
        print("此目录不存在!!!")
        return XmlList
    for x in os.listdir(fpath):
        if x.find('.xml') != -1:
            if os.path.isfile(os.path.join(fpath, x)):
                XmlList.append(x)
    return XmlList


# 遍历bin
def BrowseBin(fpath):
    binlList = []
    if not os.path.exists(fpath):
        print("此目录不存在!!!")
        return binlList
    for x in os.listdir(fpath):
        if x.find('.bin') != -1:
            if os.path.isfile(os.path.join(fpath, x)):
                binlList.append(x)
    return binlList

# 遍历目录下所有bin，包括子目录
def BrowseBins(fpath):
    binlList = []
    if not os.path.exists(fpath):
        print("此目录不存在!!!")
        return binlList
    for root, dirs, files in os.walk(fpath):
        for x in files:
            if x.find('.bin') != -1:
                binlList.append(root + os.path.sep + x)
    return binlList

# 快速排序
def quick_sort(myList, indexArr, start, end):

    try:
        if start < end:
            i, j = start, end
            base = myList[i]
            while i < j:
                while (i < j) and (myList[j] >= base):
                    j = j - 1
                myList[i] = myList[j]
                _temp = indexArr[j]
                indexArr[j] = indexArr[i]
                indexArr[i] = _temp
                while (i < j) and (myList[i] <= base):
                    i = i + 1
                myList[j] = myList[i]
                _temp = indexArr[j]
                indexArr[j] = indexArr[i]
                indexArr[i] = _temp
            myList[i] = base
            quick_sort(myList, indexArr, start, i - 1)
            quick_sort(myList, indexArr, j + 1, end)

    except:
        print('quick_sortError!!!')

