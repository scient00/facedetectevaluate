1.python3.x环境：
	import logging
	import cv2,os,string
	import numpy as np
	import datetime,pandas
	from openpyxl import *
	import matplotlib.pyplot as plt

2.测试样本说明：
	检测与标注数据存储格式：[imageName]{[number][x][y][width][height][confidence]}{...} 参考示例TestSample/***.ly
	
3.运行测试：
	python Test.py <imagePath> <labelPath> <detePath>
	
	
	
	