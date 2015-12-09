#!/usr/bin/env python

# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
from collections import Counter

# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
# beolvassa az adatokat, int64/int-ekre castolva
# ezek pedig 784-hosszu tombok:
features = np.array(dataset.data, 'int16')
# ezek sima szamok: 
labels = np.array(dataset.target, 'int')

#print(features[0])
print(dataset.data.shape)

# Extract the hog features
print('Compute features')
print('progress:')
no_features = features.shape[0]
i = 1
list_hog_fd = []
for feature in features:
	#reshape(): ez egy (28*28)-szeles array-t atmeretez 28 szeles es 28 hosszu array-re
	#hog():
	#fd: ez egy 36-hosszu array
	#
	fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
	print (fd.shape)
	list_hog_fd.append(fd)
	i+=1
	if (float(i*100)/no_features == i*100/no_features):
		print ('	' + str(i*100//no_features) + '%')
print('Done, features computed!')
# a hod_features merete: (70000, 36)
hog_features = np.array(list_hog_fd, 'float64')
print "Count of digits in dataset", Counter(labels)

# Create an linear SVM object
clf = LinearSVC()
print('Training LinearSVC')
# Perform the training
clf.fit(hog_features, labels)
print('Done, saving...')
# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)