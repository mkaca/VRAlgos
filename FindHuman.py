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

areaThreshold = 2500

cap = cv2.VideoCapture(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2()


kernel = np.ones((3,3),np.uint8)
startTime = time.time()

while(1):
    ret, frame = cap.read()
    area = 0 

    fgmask = fgbg.apply(frame)
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
   
    erosion = cv2.erode(fgmask,kernel,iterations = 1)     ## removes background crap
    
    area = np.sum(erosion.flatten()) / 255                ## Gets area of moving objects 

    if area > areaThreshold:
    	"""# find contours
    	imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    	ret,thresh = cv2.threshold(imgray,127,255,0)
    	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    	if (len(contours) > 0):
    		img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)  
    		cv2.imshow('contours',img)
    		cv2.waitKey(0)"""																			 # contours is trash!!!!!!!

    	# try using FAST!!!


    #if time.time() - startTime > 5:
    #	print( 'area:',area)
    cv2.imshow('frame',fgmask)
    cv2.imshow('areaSize',erosion)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        #plt.imshow(fgmask),plt.show()
        break

cap.release()
cv2.destroyAllWindows()