#!/usr/bin/python

from sklearn import linear_model
from sklearn.feature_selection import f_regression as f_regr
from scipy import stats
import numpy as np
import csv

y = []
X = []
labels = []

def fetch_features(): 
    global X, y, labels
    with open("features.csv","rb") as csvfile:
        reader = csv.reader(csvfile)
        for value,row in enumerate(reader):
            if value == 0: labels = row
            else:
                if len(row) < len(labels): continue
                row = [float(x) for x in row[1:]]
                y.append(row[labels.index('accuracy')])
                row.pop(labels.index('accuracy'))
                X.append(row[4:-3])

def apply_regression():
    global labels
    regr = linear_model.Ridge(alpha=0.5,normalize=True)
    print("-----------------")
    print("%8s  %7s" % ("Feature","p-value"))
    print("-----------------")
    labels = labels[5:-3]
    ret = f_regr(X,y)
    labels.pop(labels.index('accuracy'))
    for index in xrange(len(labels)):
        print "%9s: %.4f" % (labels[index], ret[1][index])

fetch_features()
apply_regression()
