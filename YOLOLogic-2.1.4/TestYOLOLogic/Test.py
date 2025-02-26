#!/usr/bin/env python

import unittest
import TestImageLoadingAndDataExtraction
import TestImageConversion

class YOLOLogicTestCase( unittest.TestCase ):
    def checkVersion(self):
        import YOLOLogic

testSuites = [unittest.makeSuite(YOLOLogicTestCase, 'test')] 

for test_type in [
            TestImageLoadingAndDataExtraction,
            TestImageConversion,
    ]:
    testSuites.append(test_type.getTestSuites('test'))


def getTestDirectory():
    try:
        return os.path.abspath(os.path.dirname(__file__))
    except:
        return '.'

import os
os.chdir(getTestDirectory())

runner = unittest.TextTestRunner()
runner.run(unittest.TestSuite(testSuites))
