import CentralPatternFind as CPF
import cv2
import time
import numpy as np 
from matplotlib import pyplot as plt
import copy


cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    #print (np.shape(frame)[0])
    ySize = np.shape(frame)[0] # height
    xSize = np.shape(frame)[1] # width
    blackArray = np.zeros((ySize,xSize,3), dtype = np.uint8)
    whiteArray = np.ones((ySize,xSize,3), dtype = np.uint8)

    # Initiate FAST object with default values                                 #### basically can use FAST or Harris for edge detection 
    fast = cv2.FastFeatureDetector_create(threshold=25)
    # find and draw the keypoints
    kp2 = fast.detect(frame,None)
    cornerImg = copy.copy(frame)
    cornerImg = cv2.drawKeypoints(cornerImg, kp2, None,color=(255,250,20))
    testImgInput1 = cv2.drawKeypoints(whiteArray, kp2, None,color=(255,250,20))

    cv2.imshow('corners', cornerImg)
    cv2.imshow('testImgInput1',testImgInput1)    # GREeeeeat for first input of central shape finder 
    #print (np.shape(cornerImg))

    k = cv2.waitKey(300) & 0xff
    if k == 27:
        #plt.imshow(fgmask),plt.show()
        break
    CPF.CPF(testImgInput1,(ySize,xSize))


cap.release()
cv2.destroyAllWindows()