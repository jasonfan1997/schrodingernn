# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:11:34 2017

@author: user98
"""

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
# Load pandas
from sklearn.model_selection import cross_val_score
import pandas as pd
import math
# Load numpy
import numpy as np
import matplotlib.pyplot as plt
DIR = "../data/stock_train_data_20170901.csv"
TESTDIR="../data/stock_test_data_20170901.csv"
COLUMNS = list(range(1,91))  #Read Feature,weight,label
all_set = pd.read_csv(DIR, skipinitialspace=True,
                             skiprows=0, usecols=COLUMNS).as_matrix()
SORT = list(range(0,89))
SORT.insert(0,89)   #89,0-87,88
all_set = all_set[:,np.array(SORT)] #Change into 0Label,Feature,88Weight
np.random.shuffle(all_set)
#training_set=all_set
#training_set=all_set
training_set=all_set
SSD=list(range(1,89))
prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
                             skiprows=0, usecols=SSD).as_matrix()
#prediction_set=all_set[math.floor(all_set.shape[0]*0.7):]             
training_weight=training_set[:,-1]
training_set=training_set[:,:-1]         
#prediction_weight=prediction_set[:,-1]
#prediction_set=prediction_set[:,:-1]    

#logreg = linear_model.LogisticRegression()
'''
SSD=list(range(1,89))
prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
                             skiprows=0, usecols=SSD).as_matrix()
'''
clf=SVC(probability=True,verbose=3)
#print(cross_val_score(clf,training_set[:,1:], training_set[:,0], cv=10))
clf.fit(training_set[:,1:],training_set[:,0])
predicted_prob=clf.predict_proba(prediction_set)
indices = pd.read_csv(TESTDIR, skipinitialspace=True, skiprows=0, usecols=[0]).as_matrix().flatten()
df = pd.DataFrame(data={'id':indices, 'proba':predictions})
df.to_csv('result_svc.csv',index=False)
print('Result saved.')