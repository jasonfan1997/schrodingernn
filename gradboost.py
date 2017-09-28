# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:14:37 2017

@author: user98
"""
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
# Load pandas
import pandas as pd
import math
# Load numpy
import numpy as np

	
DIR = "../data/stock_train_data_20170923.csv"
COLUMNS = list(range(1,91))  #Read Feature,weight,label
all_set = pd.read_csv(DIR, skipinitialspace=True,
                             skiprows=0, usecols=COLUMNS).as_matrix()
TESTDIR="../data/stock_test_data_20170923.csv"							 
SORT = list(range(0,89))
SORT.insert(0,89)   #89,0-87,88
all_set = all_set[:,np.array(SORT)] #Change into 0Label,Feature,88Weight
#np.random.shuffle(all_set)
training_set=all_set
SSD=list(range(1,89))
#prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
#                        skiprows=0, usecols=SSD).as_matrix()

training_set=all_set[0:math.floor(all_set.shape[0]*0.95)]
prediction_set=all_set[math.floor(all_set.shape[0]*0.95):]    
prediction_set=all_set[math.floor(all_set.shape[0]*0.95):]             
training_weight=training_set[:,-1]
training_set=training_set[:,:-1]         
prediction_weight=prediction_set[:,-1]
prediction_set=prediction_set[:,:-1]    

#logreg = linear_model.LogisticRegression()

def standardize_data(array):
    #takes in 2d arrays
    #relative scale
    a = array.copy()
    for i in range(a.shape[-1]):
        mean = np.mean(a[:, i])
        std = np.std(a[:, i])
        a[:,i] = (a[:,i] - mean)/std
    return a

# we create an instance of Neighbours Classifier and fit the data.
#logreg.fit(training_set[:,1:],training_set[:,0])
#predicted_class = logreg.predict(prediction_set[:,1:])
'''
clf=RandomForestClassifier(n_estimators=50000,criterion='entropy',n_jobs=-1,verbose=2,max_features="sqrt")
clf.fit(training_set[:,1:],training_set[:,0])
#predicted_class=clf.predict(prediction_set[:,1:])
predicted_proba=clf.predict_proba(prediction_set[:,1:])
#predicted_prob=clf.predict_proba(prediction_set)
los=log_loss(prediction_set[:,0],predicted_proba)
with open("RFloss.txt", "w") as output:
    output.write(str(los))

print(los)
'''
#testdata: 321674 ~ 521619
training_set=all_set
prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
                             skiprows=0, usecols=SSD).as_matrix()
prediction_set=standardize_data(prediction_set)
training_weight=training_set[:,-1]
training_data=standardize_data(training_set[:,1:-1])
labels=training_set[:,0]
clf=GradientBoostingClassifier(n_estimators=200, verbose=1)
clf.fit(training_data,labels)    
predictions=clf.predict_proba(prediction_set)

indices = pd.read_csv(TESTDIR, skipinitialspace=True, skiprows=0, usecols=[0]).as_matrix().flatten()
df = pd.DataFrame(data={'id':indices, 'proba':predictions[:,1]})
df.to_csv('result_gradboost2.csv',index=False)
print('Result saved.')
