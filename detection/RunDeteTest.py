#!/usr/bin/env python
# -*- coding: utf-8 -*-


from BasicMethod.BasicMethod import *
import datetime,logging
from detection.FaceDataProcess import readConfigFile, readTestData
from detection.FaceDeteEvaluate import DetectEvaluate
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

def detectTest(imgpath,labelpath,detepath,confpath = 'config'):
    try:
        confile = readConfigFile(confpath + '/config.yaml' )
        vLabelList = readTestData(labelpath,confile.getClasses()[0])
        vDetectList = readTestData(detepath, confile.getClasses()[0])

        startTime = datetime.datetime.now()
        deteTest = DetectEvaluate(confile.getClasses())
        deteTest.set(vDetectList, vLabelList, imgpath)
        deteTest.showInfo(SaveImage = True,ShowImage = False)
        deteTest.evaluate()
        deteTest.generatePreRecCurve('TestResult',confile.getIouValue())
        deteTest.generateExcelReports('TestResult', confile.getIouValue())
        deteTest.generateCsvReports('TestResult',confile.getIouValue())
        deteTest.generateHtmlReports('TestResult',confile.getIouValue())
        endTime = datetime.datetime.now()
        print('evaluateTime=',(endTime - startTime).total_seconds()*1000,'ms')

    except:
        logging.error('detectTest')
