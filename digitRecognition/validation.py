#!/usr/bin/env python

# Import the modules
import cv2
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter
from os import listdir
from os.path import isfile, join

# Load the classifier
clf = joblib.load("digits_cls.pkl")

mypath = '/home/balint/workspace/svm/digitRecognition/val'
paths = [f for f in listdir(mypath) if isfile(join(mypath, f))]

list_hog_fd = []
labels = []
predictions = []
files = []
no_images = len(paths)

for path in paths:
	# a kep beolvasasa
	im = cv2.imread(mypath + '/' + path)
	# label meghatarozasa a fajlnev alapjan 
	label = int(path[-10:-8])-1 
	# a kep atalakitasa szurkearanyalatossa
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# gaussian (http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#gaussian-filtering)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
	# binaris inverse threshold azaz fekete-feher keppe alaktasa
	ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
	# a szamjegy korbehatarolasa
	ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# a megtalalt - remelhetoleg egy - szamjegy befoglalo szogszoge 
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	# elunk a feltetelezessel, hogy egyet talalt meg, de azert a biztonsag kedveert a 4*4-nel kisebb teglalapokat eldobjuk
	for rect in rects:
		if ( rect[2] > 4 and rect[3] > 4  ):
			break;	
	# az uniformizalas miatt kicsit kitagitjuk a befoglalo teglalapot
	leng = int(rect[3] * 1.05)
	pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
	pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
	roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
	# a megkapott teglalapot atalakitjuk 28*28-as meretre
	roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
	# megvastagitjuk a szamjegyeket 
	roi = cv2.dilate(roi, (3, 3))
	# kiszamoljuk a hog gradienseket
	fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	nbr = clf.predict(np.array([fd], 'float64'))
	labels.append(label)
	predictions.append(nbr)
	files.append(path)

missed_msg = ''
no_missed = 0
for i in range(len(labels)):
	if (  str(labels[i]) != str(predictions[i])[1:2]  ):
		missed_msg = '!!!!!!!'
		no_missed +=1		
	print(str(files[i]) + '\t actual:' + str(labels[i]) + '\t predicted:' + str(predictions[i])[1:2] + '\t' + missed_msg)
	missed_msg = ''

print ('Hibas oszalyozasok szama: ' + str(no_missed))

cv2.waitKey()