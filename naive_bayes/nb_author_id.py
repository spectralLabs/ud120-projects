#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

# get the Gaussian naive bayes classifier object
from sklearn.naive_bayes import GaussianNB 
# import the accuracy module
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# create the classifier object
clf = GaussianNB()

# start the timer
t0 = time()

# fit it to the data / train it on the data
clf.fit(features_train, labels_train)

# print the training time
print "training time:", round(time() - t0, 3), "s"

# start the timer again for getting prediction time
t1 = time()

# get a prediction on some data
pred = clf.predict(features_test)

# print the prediction time
print "prediction time:", round(time() - t1, 3), "s"
# get the accuracy of the prediction
accuracy = accuracy_score(labels_test, pred)
#########################################################

# print the accuracy to see it
print accuracy
