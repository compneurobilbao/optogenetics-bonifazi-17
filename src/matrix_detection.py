# -*- coding: utf-8 -*-

import numpy as np
import cv2

img = cv2.imread('./data/raw/1.jpg')
gray = cv2.imread('./data/raw/1.jpg',0)

ret,thresh = cv2.threshold(gray,127,255,1)

contours, h = cv2.findContours(thresh,1,2)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print(len(approx))
    if len(approx)==5:
        print("pentagon")
        cv2.drawContours(img,[cnt],0,255,-1)
    elif len(approx)==3:
        print("triangle")
        cv2.drawContours(img,[cnt],0,(0,255,0),-1)
    elif len(approx)==4:
        print("square")
        cv2.drawContours(img,[cnt],0,(0,0,255),-1)
    elif len(approx) == 9:
        print("half-circle")
        cv2.drawContours(img,[cnt],0,(255,255,0),-1)
    elif len(approx) > 15:
        print("circle")
        cv2.drawContours(img,[cnt],0,(0,255,255),-1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


http://opencvpython.blogspot.com.es/2012/06/sudoku-solver-part-2.html
http://stackoverflow.com/questions/11424002/how-to-detect-simple-geometric-shapes-using-opencv
https://dsp.stackexchange.com/questions/3595/finding-squares-in-image



import cv2
import numpy as np

img = cv2.imread('./data/raw/1.jpg')
gray = cv2.imread('./data/raw/1.jpg',cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


biggest = None
max_area = 0
for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area
                        
cv2.drawContours()       




cv2.imwrite('test.png',gray)




                 