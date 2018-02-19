"""

         1. Feature that detects humans
            - based on movement and shapes.
            - based on walking pattern
            	-Feature: detects when movement is abnormal (drunk ,dizzy, tired, sick, etc...)
            	-Feature: can select area that is unsafe and see if the person is in an area that is safe or nah  (green box vs red box ting)
            	-Feature: can look for particular things like lockout locks in place, to prevent people from locking themselves out and stuffz
            - USEFUL FOR: power plants, toyota and tesland safety and such... # SAFETYYYY..... # heavy machinery
            - Can integrate with PLCs for easy installation

		a. Find movement over x-threshold
		b. Find moving object size....
			i. For example: if hand is moving, extrapolate the rest of the torso to find the human torso
					See if torso matches torso hardcoded default shape, and then extrapolate legs and arms
		c. Use Color, texture, edges, corners
	    d. Get pixels that were moving, and use those areas to find corners and edges and contours
"""
import cv2
import time
import numpy as np 
from matplotlib import pyplot as plt
import copy

areaThreshold = 2500

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('videoplayback.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2()


kernel = np.ones((3,3),np.uint8)
startTime = time.time()

while(1):
    ret, frame = cap.read()
    area = 0 

    ## gets moving objects in frame
    fgmask = fgbg.apply(frame)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
   
    erosion = cv2.erode(fgmask,kernel,iterations = 1)     ## removes background crap
    
    area = np.sum(erosion.flatten()) / 255                ## Gets area of moving objects 

    if area > areaThreshold:


    	# Initiate FAST object with default values                                 #### basically can use FAST or Harris for edge detection 
    	fast = cv2.FastFeatureDetector_create(threshold=25)
    	# find and draw the keypoints
    	kp = fast.detect(erosion, None)
    	img2 = copy.copy(erosion)
    	img2 = cv2.drawKeypoints(img2, kp, None,color=(255,0,0))

    	## Edge detection
    	#             gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    	#gray = np.float32(gray)
    	dst = cv2.cornerHarris(erosion,2,3,0.04)
    	#result is dilated for marking the corners, not important
    	dst = cv2.dilate(dst,None)
    	# Threshold for an optimal value, it may vary depending on the image.
    	frame[dst>0.01*dst.max()]=[0,0,255]
    	### Which is basically the same as :
    	ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)

    #if time.time() - startTime > 5:
    #	print( 'area:',area)
    cv2.imshow('fgmask',fgmask)
    cv2.imshow('erosion',erosion)
    cv2.imshow('fast',img2)
    cv2.imshow('dst',dst)                                                    ##### this is pretty good tbh .... finding harrison on the moving object
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        #plt.imshow(fgmask),plt.show()
        break

cap.release()
cv2.destroyAllWindows()