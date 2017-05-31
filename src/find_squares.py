#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:20:31 2017

@author: https://github.com/abidrahmank/OpenCV2-Python/blob/master/OpenCV_Python_Blog/sudoku_v_0.0.6/sudoku.py
"""

import cv2
import numpy as np
import time, sys


def rectify(h):
		''' this function put vertices of square we got, in clockwise order '''
		h = h.reshape((4,2))
		hnew = np.zeros((4,2),dtype = np.float32)

		add = h.sum(1)
		hnew[0] = h[np.argmin(add)]
		hnew[2] = h[np.argmax(add)]
		
		diff = np.diff(h,axis = 1)
		hnew[1] = h[np.argmin(diff)]
		hnew[3] = h[np.argmax(diff)]

		return hnew



img = cv2.imread('./data/raw/1.jpg')
gray = cv2.imread('./data/raw/1.jpg', cv2.CV_8UC1)


thresh = cv2.adaptiveThreshold(gray,255,1,1,5,2)
im2, contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_area = gray.size	# this is area of the image

for i in contours:
	if cv2.contourArea(i)> image_area/2: # if area of box > half of image area, it is possibly the biggest blob
		peri = cv2.arcLength(i,True)
		approx = cv2.approxPolyDP(i,0.02*peri,True)
		#cv2.drawContours(img,[approx],0,(0,255,0),2,cv2.CV_AA)
		break


h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)	# this is corners of new square image taken in CW order

approx = rectify(approx)	# we put the corners of biggest square in CW order to match with h

retval = cv2.getPerspectiveTransform(approx,h)	# apply perspective transformation
warp = cv2.warpPerspective(img,retval,(450,450))  # Now we get perfect square with size 450x450

warpg = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)	# kept a gray-scale copy of warp for further use



sudo = np.zeros((8,8),np.uint8)		# a 9x9 matrix to store our sudoku puzzle

denoised = cv2.fastNlMeansDenoising(warpg, h=10)
thresh = cv2.threshold(denoised,20,255,0)

#
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
#erode = cv2.erode(thresh[1], kernel, iterations =100)
#dilate =cv2.dilate(erode,kernel,iterations =100)
cv2.imwrite('test.png',cv2.drawContours(im3, contours, -1, (25, 255, 0), 1))

im3, contours, hierarchy = cv2.findContours(thresh[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.imwrite('test.png',cv2.drawContours(im3, contours, -1, (25, 25, 0), 5))





for cnt in contours:
    area = cv2.contourArea(cnt)
    if area>2:
        print('hey')
	
		(bx,by,bw,bh) = cv2.boundingRect(cnt)
		if (100<bw*bh<1200) and (10<bw<40) and (25<bh<45):
			roi = dilate[by:by+bh,bx:bx+bw]
			small_roi = cv2.resize(roi,(10,10))
			feature = small_roi.reshape((1,100)).astype(np.float32)
			ret,results,neigh,dist = model.find_nearest(feature,k=1)
			integer = int(results.ravel()[0])
			
			gridy,gridx = (bx+bw/2)/50,(by+bh/2)/50	# gridx and gridy are indices of row and column in sudo
			sudo.itemset((gridx,gridy),integer)
            
sudof= sudo.flatten()
strsudo = ''.join(str(n) for n in sudof)
















