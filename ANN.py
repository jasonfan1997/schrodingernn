# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 16:14:11 2017

@author: user98
"""
######CSV Style: 0:ID 1-88:feature 89:weight 90:label
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


#Generate Training Set
DIR = "../data/stock_train_data_20170901.csv"
COLUMNS = list(range(1,91))  #Read Feature,weight,label
all_set = pd.read_csv(DIR, skipinitialspace=True,
                             skiprows=0, usecols=COLUMNS).as_matrix()
SORT = list(range(0,89))
SORT.insert(0,89)   #89,0-87,88
all_set = all_set[:,np.array(SORT)] #Change into 0Label,Feature,88Weight
#np.random.shuffle(all_set)
training_set=all_set
	             
	             
TESTDIR="../data/stock_test_data_20170901.csv"

TRAINING_STEPS =5000
LEARNING_RATE = 0.002

MODEL_DIR = "../data/model1"

BATCH_SIZE = 800
OPTIMIZER = "Adam"

#predicted_result = None
exp = None
predicted_prob = None
prediction_set = None
predicted_class = None

bias_3 = None
weight_3 = None


n1= 88   
n2=44

n3= 2
n4= 22
n5= 2

'''def normalize(a):
    a_norm = tf.norm(a,axis=1,keep_dims=True)
    a_normalized = tf.divide(a,a_norm)
    return a_normalized'''
    

def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

    #Build the network
  #input_layer = tf.contrib.layers.input_from_feature_columns(columns_to_tensors=features, feature_columns=feature_cols)
  '''
  init3_n1 = tf.constant_initializer(np.eye(n1, M=3))
  init_3 = tf.constant_initializer(np.eye(3, M=n3))
  init_iden = tf.constant_initializer(np.eye(n1, M=n3))
  init3 = tf.constant_initializer(np.eye(n3, M=n2))'''
  
  
  # Comy_estimatorect the first hidden layer to input layer
  first_hidden_layer = tf.layers.dense(features, n1, activation=tf.nn.relu)
  
  first_processed = tf.contrib.layers.dropout(
         
                  first_hidden_layer, keep_prob=1)


  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(first_processed, n2, activation=tf.nn.relu)
  
  second_processed = tf.contrib.layers.dropout(
          #tf.contrib.layers.layer_norm(
                  second_hidden_layer,1
                                        )  
  
  third_hidden_layer = tf.layers.dense(second_processed, n3, activation=tf.nn.relu)
  
  third_processed = tf.contrib.layers.dropout(
           #tf.contrib.layers.layer_norm(,activation_fn=)
                       third_hidden_layer,1)
  '''
  fouth_hidden_layer = tf.layers.dense(third_processed, n4, activation=tf.nn.relu)
  
  fouth_processed = tf.contrib.layers.dropout(
           #tf.contrib.layers.layer_norm(,activation_fn=)
                       fouth_hidden_layer, 0.9) 
  fifth_hidden_layer = tf.layers.dense(fouth_processed, n4, activation=tf.nn.relu)
  
  fifth_processed = tf.contrib.layers.dropout(
           #tf.contrib.layers.layer_norm(,activation_fn=)
                       fifth_hidden_layer, 1) 
  '''
  # Comy_estimatorect the output layer to second hidden layer (no activation fn)
  logits = tf.layers.dense(third_processed, 2, activation=None)
  
  weights = tf.constant(params["weights"])
  #logits = tf.contrib.layers.layer_norm(pre_logits,activation_fn=None)
  
  #logits_reshaped = tf.reshape(logits, [-1, 3])
  
  #logits = tf.contrib.layers.unit_norm(pre_logits, 1, epsilon=1e-20)
  
  #softmax_probability = tf.nn.softmax(logits)
  
  #normalized prob
  #normalized_prob = 
  
  
  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Calculate loss
  onehot_labels = tf.reshape(tf.contrib.layers.one_hot_encoding(targets, 2),[-1, 2])
  

    
  #loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, weights=weights)
  loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)


  loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, weights=weights)
  

  # Calculate Loss (for both TRAIN and EVAL modes)
  '''if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
     
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="Adam")'''

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer= OPTIMIZER) 
      
  # Return a ModelFnOps object (eval_metrics not included)
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def input_fn(data_set):
  '''feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  #features = tf.constant([data_set[k].values for k in FEATURES])
  labels = tf.constant(data_set[LABEL].values)'''
  
  features = tf.constant(np.delete(data_set, 0, 1))
  labels = tf.constant(np.int_(np.delete(data_set, np.s_[1:], 1)))
  return features, labels

def new_input_fn(data_set):
  '''feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  #features = tf.constant([data_set[k].values for k in FEATURES])
  labels = tf.constant(data_set[LABEL].values)'''
  
  features = tf.constant(data_set)
  labels = tf.constant(np.int_(np.delete(data_set, np.s_[1:], 1)))
  return features, labels


def new_input_fn(data_set):
  '''feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  #features = tf.constant([data_set[k].values for k in FEATURES])
  labels = tf.constant(data_set[LABEL].values)'''
  
  features = tf.constant(data_set)
  labels = tf.constant(np.int_(np.delete(data_set, np.s_[1:], 1)))
  return features, labels

def main():
  # Load datasets

  #skip some rows (use them as test/pred set later) 
  #not_load = np.random.randint(1000, size=10)
  global prediction_set
  global training_weight
  

	             
	             


  training_weight=training_set[:,-1]
  training_set=training_set[:,:-1]
  SSD=list(range(1,89))
  prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
                             skiprows=0, usecols=SSD).as_matrix()
	             
  

  '''
  training_set=all_set[0:math.floor(all_set.shape[0]*0.7)]
  prediction_set=all_set[math.floor(all_set.shape[0]*0.7):]
  '''
  training_weight=training_set[:,-1]
  training_set=training_set[:,:-1]
  SSD=list(range(1,89))
  prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
                             skiprows=0, usecols=SSD).as_matrix()

  #Prediction set without HP column, used to calc expectation                             

    # Feature cols
  
  model_params = {"learning_rate": LEARNING_RATE, "model_dir": MODEL_DIR, "weights": training_weight}
  configs = tf.contrib.learn.RunConfig(save_summary_steps=500)


  my_estimator = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params, 
                                            config=configs, 
                                            model_dir= MODEL_DIR)

  
  
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=lambda: input_fn(training_set),
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

  #Initialize the training!!!
  my_estimator.fit(input_fn=lambda: input_fn(training_set), steps=TRAINING_STEPS)
  
  
  #SKCompat Version (accepts using batch size)
  ''''''
  x = np.delete(training_set, 0, 1)
  y = np.int_(np.delete(training_set, np.s_[1:], 1))'''
  
  
  '''
  my_estimator = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)
  my_estimator.fit(x, y , steps=TRAINING_STEPS, batch_size=BATCH_SIZE, monitors=[validation_monitor])
 
'''

  global predicted_result
  global exp
  global predicted_prob
  
  #Removed the outside "list"
  predicted_result = my_estimator.predict(input_fn=lambda: new_input_fn(prediction_set),as_iterable=False)
  predicted_prob = predicted_result["probabilities"]
  predicted_class = predicted_result["classes"]

  

  np.save('result.npy',predicted_prob)
  np.savetxt('result.csv',predicted_prob,delimiter=',')

    
  '''
  global bias_3
  global weight_3
  
  print(my_estimator.get_variable_names())
  #bias_3 = my_estimator.get_variable_value('fully_connected_3/biases')
  weight_3 = my_estimator.get_variable_value('dense/kernel')'''
  
  '''
  exp = np.multiply(predicted_prob, np.array(prediction_set2))
  
  print(exp)
  
  '''
  def accuracy():
      
      count = 0
      correct = 0
      for i in range(0,predicted_class.shape[0]):
          count += 1
          if predicted_class[i] == prediction_set[i, 0]:
              correct += 1
      acc = correct/count
      print("Accuracy: " + str(acc))
      print("Total matches: " + str(count)) 
      count = 0
      correct = 0
      for i in range(0,predicted_class.shape[0]):
          if prediction_set[i, 0]==1:
              count += 1
          if predicted_class[i] == prediction_set[i, 0] and prediction_set[i, 0]==1:
              correct += 1
      acc = correct/count
      print("PSR Accuracy: " + str(acc))
      print("Total matches: " + str(count)) 
      count = 0
      correct = 0
      for i in range(0,predicted_class.shape[0]):
          if prediction_set[i, 0]==0:
              count += 1
          if predicted_class[i] == prediction_set[i, 0] and prediction_set[i, 0]==0:
              correct += 1
      acc = correct/count
      print("AGN Accuracy: " + str(acc))
      print("Total matches: " + str(count)) 
      
  def newaccuracy():
      psrcorrect=0
      psrwrong=0
      agncorrect=0
      agnwrong=0
      for i in range(0,predicted_class.shape[0]):
          if predicted_class[i]==0 and prediction_set[i, 0]==0:
              agncorrect+=1
          if predicted_class[i]==0 and prediction_set[i, 0]==1:
              agnwrong +=1
          if predicted_class[i]==1 and prediction_set[i, 0]==1:
              psrcorrect+=1
          if predicted_class[i]==1 and prediction_set[i, 0]==0:
              psrwrong +=1
      print(agncorrect)
      print(agnwrong)
      print(psrcorrect)
      print(psrwrong)
      
  
  '''
  def profit_rate():
      count = 0
      profit = 0
      for i in range(0,exp.shape[0]):
          if np.max(exp[i,:])>1.1:
              count += 1
              if np.argmax(exp[i,:]) == prediction_set[i, 0]:
                      profit += prediction_set2.iloc[i,np.argmax(exp[i,:])]
      pr = profit/count
      print("Profit rate: " + str(pr))
      print("Count: " + str(count))
      

  def profit_rate_2():
      put = 0
      profit = 0
      count = 0
      def invest(multiple):
            nonlocal count
            nonlocal profit
            nonlocal put
            count += 1
            put += multiple
            if np.argmax(exp[i,:]) == prediction_set[i,0]:
                      profit += multiple * prediction_set2.iloc[i,np.argmax(exp[i,:])]

      for i in range(0,exp.shape[0]):
          if np.max(exp[i,:])>1.3:
              invest(3)
          elif np.max(exp[i,:])>1.2:
              invest(2)
          elif np.max(exp[i,:])>1.1:
              invest(1)
              
      pr = profit/put  
      
      print("Profit rate: " + str(pr))
      print("Count: " + str(count)) 
        
  #profit_rate()
  #profit_rate_2() 
  '''
  #newaccuracy()
  
  '''
  a = tf.reshape(tf.constant(float(input())),[1,None])
  b = tf.reshape(tf.constant(float(input())),[1,None])
  c = tf.reshape(tf.constant(float(input())),[1,None])
  def my_input_fn():
      labels = None
      input_dict = {'AVG_H': a, 'AVG_D': b, 'AVG_A': c}
      return input_dict, labels
  
  input_pro = list(my_estimator.predict_proba(input_fn=my_input_fn
                    , as_iterable=False))
  '''

if __name__ == "__main__":
    main()