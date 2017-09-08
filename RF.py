# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:14:37 2017

@author: user98
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
# Load pandas
import pandas as pd
import math
# Load numpy
import numpy as np
DIR = "../data/stock_train_data_20170901.csv"
COLUMNS = list(range(1,91))  #Read Feature,weight,label
all_set = pd.read_csv(DIR, skipinitialspace=True,
                             skiprows=0, usecols=COLUMNS).as_matrix()
SORT = list(range(0,89))
SORT.insert(0,89)   #89,0-87,88
all_set = all_set[:,np.array(SORT)] #Change into 0Label,Feature,88Weight
np.random.shuffle(all_set)
#training_set=all_set
#training_set=all_set
training_set=all_set[0:math.floor(all_set.shape[0]*0.7)]
prediction_set=all_set[math.floor(all_set.shape[0]*0.7):]    
#prediction_set=all_set[math.floor(all_set.shape[0]*0.7):]             
training_weight=training_set[:,-1]
training_set=training_set[:,:-1]         
prediction_weight=prediction_set[:,-1]
prediction_set=prediction_set[:,:-1]    
TESTDIR="../data/stock_test_data_20170901.csv"
#logreg = linear_model.LogisticRegression()
'''
SSD=list(range(1,89))
prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
                             skiprows=0, usecols=SSD).as_matrix()
'''
# we create an instance of Neighbours Classifier and fit the data.
#logreg.fit(training_set[:,1:],training_set[:,0])
#predicted_class = logreg.predict(prediction_set[:,1:])
clf=RandomForestClassifier(n_estimators=10000,criterion='gini')
clf.fit(training_set[:,1:],training_set[:,0])
predicted_class=clf.predict(prediction_set[:,1:])
predicted_proba=clf.predict_proba(prediction_set[:,1:])
def entropy():
    entropy=0
    for i in range(0,predicted_class.shape[0]):
        entropy-=prediction_set[i, 0]*np.log(predicted_proba[])
