#!/usr/bin/env python

# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image
#im = cv2.imread("img002-053.png")
im = cv2.imread("sajat_01.jpg")
#im = cv2.imread("/home/balint/Desktop/English/Hnd/Img/Sample010/img010-054.png")
# im.shape: (height, width, 3[rgb]) 
print('This image is ' + str(im.shape[0]) + '*' + str(im.shape[1]) + ' pixel and has ' + str(im.shape[2]) + ' color values.')
# Convert to grayscale and apply Gaussian filtering (http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#gaussian-filtering)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# gaussian filtering: 
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
# ret if a float, we neves use that.
# im_th is a (height, width) array

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# ctrs is list of object vertex
# hier(archy) is an array of the found rects, we neves use that

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
# rects is an (foundObjectNo, 4) array, and the second is 
#	top-left corner coordinates and heigh, width

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.

for rect in rects:
	#rect = rects[1]
	if (rect[2] <= 4 or rect[3] <= 4):
		continue
	# Draw the rectangles to two diagonal corner, with color, with thickness 
	cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2) 
	# Make the rectangular region around the digit
	leng = int(rect[3] * 1.05)
	pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
	pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
	roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
	# Resize the image, ROI = rectangular of interest
	roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
	# megvastagitjuk a szamot
	roi = cv2.dilate(roi, (3, 3))
	# Calculate the HOG features
	roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), visualise=False)
	nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
	# cv2.putText(img, text, origin, font, fontScale, color, thickness)
	cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 20, 20), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.imwrite('output.png',im)
cv2.waitKey()