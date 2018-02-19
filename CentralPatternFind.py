


## input : pixels of desire (from fast)                                DONE
           #image size                                                 DONE
""" output: largest shape in the middle (cluster of points)
            and size of middle shape in pixels (height and width)
  magic: a. focus on central part of window ( see book)                 DONE
         b. exclude outlier clusters and choose cluster with:

                find closest point to the center
                choose X amoutn of points closest to it
                If abs((median of distaneces) - (mean of distances)) > thresholdY, then discard furthest point to make amoutn of points (X-1) and try again... keep doing until thresholdY is met. 
                       If cluster size gets below N due to the reduction algo, then try using a different point in one of the 'corners' of the central part of window, and repeat algo

                optional params: X, ThresholdY, N, 

        c. Once cluster is chosen, identify its size and do some mods to it like erosion and dilution to get general cluster shape (# of sides and corners) ...which will be used for future frames --> for identifying distance change!!   

        PROBLEMS:    the shape has to be consistent in between frames..there are slight variations from interest points from frame to frame (can get median of frames or seomething like that)
"""
import cv2 
import numpy as np 

class CPF(object):
	# inputImg type : points of interest on white background, size: (y,x,3)
	# size type:  ySize (height), xSize (width)
	def __init__(self, inputImg, size):
		#init params
		self.inputImg = inputImg
		self.size = size
		if (int(np.shape(inputImg)[0]) != int(size[0]) or int(np.shape(inputImg)[1]) != size[1]):
			raise ValueError('size does not match image...imageSize: %i,%i and sizeInputted: %i,%i'%(np.shape(inputImg)[0],np.shape(inputImg)[1],size[0],size[1]))
		self.getCentralWindow()

	def getCentralWindow(self, proportionFactorEdgeTrim = 4):  # for example, the PFET = 4 trims 1/4 of the window size from all directions, PFET = 6 trims 1/6 of window size from all directions
		ySize = self.size[0]
		xSize = self.size[1]
		yFrag = int(self.size[0] / proportionFactorEdgeTrim)
		xFrag = int(self.size[1] / proportionFactorEdgeTrim)
		croppedImg = self.inputImg[yFrag:ySize-yFrag,xFrag:xSize-xFrag]
		cv2.imshow('hiiii',croppedImg)
		cv2.waitKey(0)


