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

# fit it to the data / train it on the data
clf.fit(features_train, labels_train)

# get a prediction on some data
pred = clf.predict(features_test)

# get the accuracy of the prediction
accuracy = accuracy_score(labels_test, pred)
#########################################################

# print the accuracy to see it
print accuracy
