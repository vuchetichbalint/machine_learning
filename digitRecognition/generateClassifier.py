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

mypath = '/home/balint/workspace/svm/digitRecognition/data'
paths = [f for f in listdir(mypath) if isfile(join(mypath, f))]

list_hog_fd = []
labels = []
i = 1
no_images = len(paths)

print('Compute features')
print('progress:')

for path in paths:
	# a kep beolvasasa
	print('		' + mypath + '/' + path)
	im = cv2.imread(mypath + '/' + path)
	# label meghatarozasa a fajlnev alapjan 
	label = int(path[-10:-8])-1 
	# a kep atalakitasa szurkearanyalatossa
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# gaussian blur (http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#gaussian-filtering)
	im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
	# binaris inverse threshold azaz fekete-feher keppe alaktasa
	ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
	# a szamjegy korbehatarolasa
	ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# a megtalalt - remelhetoleg egy - szamjegy befoglalo szogszoge 
	rects = [cv2.boundingRect(ctr) for ctr in ctrs]
	# elunk a feltetelezessel, hogy egyet talalt meg
	rect = rects[0]
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
	# hozzafuzzuk az eddigiekhez 
	list_hog_fd.append(fd)
	labels.append(label)
	# progress kiirasa
	if (float(i*100)/no_images == i*100/no_images):
		print ('	' + str(i*100//no_images) + '%')
	i+=1

# atalakitas NumPy formatumra
hog_features = np.array(list_hog_fd, 'float64')
labels = np.array(labels, 'int')

print('Training LinearSVC')
clf = LinearSVC()
print('Done, saving...')
clf.fit(hog_features, labels)
joblib.dump(clf, "digits_cls.pkl", compress=3)