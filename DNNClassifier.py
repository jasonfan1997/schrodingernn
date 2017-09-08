#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""DNNRegressor with custom input_fn for Housing dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

FEATURES = []
for i in range(0,88):
    FEATURES.append("feature" + str(i))
WITH_WEIGHT = FEATURES.copy()
WITH_WEIGHT.append("weight")
COLUMNS=WITH_WEIGHT.copy()
COLUMNS.append("label")
LABEL = "label"


def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in WITH_WEIGHT}
  feature_cols["weight"] = tf.reshape(feature_cols["weight"], shape=(-1, 1))
  labels = tf.constant(data_set[LABEL].values)
  labels = tf.reshape(labels, shape=(-1, 1))
  return feature_cols, labels

def original_input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  #labels = tf.constant(data_set[LABEL].values)
  labels = None
  return feature_cols, labels


def main():
  # Load datasets
  PRED_DIR="../data/stock_test_data_20170901.csv"
  TRAIN_DIR = "../data/stock_train_data_20170901.csv"
  training_set = pd.read_csv(TRAIN_DIR, skipinitialspace=True,
                             skiprows=0, usecols=COLUMNS)
  #test_set = pd.read_csv("./EPL_1617_ALL.csv", skipinitialspace=True, skiprows=0, usecols=COLUMNS)

  prediction_set = pd.read_csv(PRED_DIR, skipinitialspace=True,
                               skiprows=0, usecols=FEATURES)
  # Feature cols
  feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.

  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                              hidden_units=[40, 20],
                                              n_classes=2,
                                              activation_fn=tf.nn.relu,
                                              dropout=0.3,
                                              weight_column_name="weight",
                                              model_dir="./test2",
                                              optimizer=tf.train.AdamOptimizer)

  '''old   validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50)'''


  # Fit model.
  classifier.fit(input_fn=lambda: input_fn(training_set), steps=400) 
  
  
  
  '''  regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[10, 10],
                                            model_dir="/tmp/epl_model")

  # Fit  old
  regressor.fit(input_fn=lambda: input_fn(training_set), steps=3000)'''

  
  '''  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y
    '''
  '''
  # Score accuracy
  ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions
  y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
  # .predict() returns an iterator; convert to a list and print predictions
  predictions = list(itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

  '''
  '''
  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=lambda: input_fn(prediction_set),
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
  '''  
  
 
  #predictions = list(classifier.predict(input_fn=lambda: input_fn(prediction_set)))
  predicted_prob = np.array(classifier.predict_proba(input_fn=lambda: original_input_fn(prediction_set), as_iterable=False))
  np.save('result.npy', predicted_prob)
  '''
  print(prediction_set)
  
  exp = np.multiply(np.array(predicted_prob),np.array(prediction_set2))
  
  print(predicted_prob)
  print(exp)
  
  np.argmax(exp, axis=0)
  
  def accuracy():
      count = 0
      correct = 0
      for i in range(0,exp.shape[0]):
          count += 1
          if predicted_class[i] == prediction_set[i, 0]:
              correct += 1
      acc = correct/count
      print("Accuracy: " + str(acc))
      print("Total matches: " + str(count))  
      
  def profit_rate():
      count = 0
      profit = 0
      for i in range(0,exp.shape[0]):
          if np.max(exp[i,:])>1.1:
              count += 1
              if np.argmax(exp[i,:]) == prediction_set.iloc[i,0]:
                      profit += prediction_set2.iloc[i,prediction_set.iloc[i,0]]
      pr = profit/count                
      print("Profit rate: " + str(pr))
      print("Count: " + str(count)) 

  def profit_rate():
      count = 0
      profit = 0
      for i in range(0,exp.shape[0]):
          if np.max(exp[i,:])>1:
              count += 1
              if np.argmax(exp[i,:]) == prediction_set.iloc[i,0]:
                      profit += prediction_set2.iloc[i,prediction_set.iloc[i,0]]
      pr = profit/count                
      print("Profit rate: " + str(pr))
      print("Count: " + str(count))     

  def profit_rate_2():
      put = 0
      profit = 0
      count = 0
      
      for i in range(0,exp.shape[0]):
          if np.max(exp[i,:])>1.3:
              count += 0
              put += 0
              
              if np.argmax(exp[i,:]) == prediction_set.iloc[i,0]:
                      profit += 0 * prediction_set2.iloc[i,prediction_set.iloc[i,0]]
          elif np.max(exp[i,:])>1.15:
              count += 1
              put += 2
              if np.argmax(exp[i,:]) == prediction_set.iloc[i,0]:
                      profit += 2 * prediction_set2.iloc[i,prediction_set.iloc[i,0]]
          elif np.max(exp[i,:])>1.05:
              count += 1
              put += 1
              if np.argmax(exp[i,:]) == prediction_set.iloc[i,0]:
                      profit += 1 * prediction_set2.iloc[i,prediction_set.iloc[i,0]] 
         
      pr = profit/put  
      
      print("Profit rate: " + str(pr))
      print("Count: " + str(count)) 
  '''
        
  '''def profit_rate():
      count = 0
      profit = 0
      for i in range(0,exp.shape[0]):
          if np.max(exp[i,:])>1:
              count += 1
              if np.argmax(exp[i,:]) == prediction_set.iloc[i,0]:
                      profit += prediction_set2.iloc[i,prediction_set.iloc[i,0]]
      pr = profit/count                
      print("Profit rate: " + str(pr))    '''  
      
        
  #profit_rate()
  #profit_rate_2() 
  
  
  
  '''print(
      "Predictions:    {}\n"
      .format(predictions))
  
  print(
      "Predictions:    {}\n"
      .format(predicted_prob))'''
  
if __name__ == "__main__":
    main()