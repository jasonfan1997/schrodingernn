# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:00:41 2017

@author: user98
"""

from sklearn.neighbors import KNeighborsClassifier
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

def standardize_data(array):
    #takes in 2d arrays
    #relative scale
    a = array.copy()
    for i in range(a.shape[-1]):
        mean = np.mean(a[:, i])
        std = np.std(a[:, i])
        a[:,i] = (a[:,i] - mean)/std
    return a

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
clf=KNeighborsClassifier(n_neighbors=20,n_jobs=-1)
clf.fit(standardize_data(training_set[:,1:]),training_set[:,0])
predicted_prob=clf.predict_proba(standardize_data(prediction_set))

#testdata: 321674 ~ 521619
indices = pd.read_csv(TESTDIR, skipinitialspace=True, skiprows=0, usecols=[0]).as_matrix().flatten()
df = pd.DataFrame(data={'id':indices, 'proba':predicted_prob[:,1]})
df.to_csv('result_notstan_duo.csv',index=False)
print('Result saved.')