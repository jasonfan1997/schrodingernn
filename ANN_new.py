# -*- coding: utf-8 -*-
######CSV Style: 0:ID 1-88:feature 89:weight 90:label
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib import learn

tf.logging.set_verbosity(tf.logging.INFO)


#Generate Training Set
DIR = "../data/stock_train_data_20170901.csv"
COLUMNS = list(range(1,91))  #Read Feature,weight,label
all_set = pd.read_csv(DIR, skipinitialspace=True,
                             skiprows=0, usecols=COLUMNS).as_matrix()
SORT = list(range(0,89))
SORT.insert(0,89)   #89,0-87,88
all_set = all_set[:,np.array(SORT)] #Change into 0Label,Feature,88Weight
np.random.shuffle(all_set)
#training_set=all_set
training_set=all_set[0:math.floor(all_set.shape[0]*0.7)]
validation_set=all_set[math.floor(all_set.shape[0]*0.7):] 

#Generate Testing numbers and training weight	
TESTDIR="../data/stock_test_data_20170901.csv"

pred_col=list(range(1,89))   #1-88,Features
prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
                             skiprows=0, usecols=pred_col).as_matrix()

	

	
#Training Parameters
MODEL_DIR = "../data/model1"
TRAINING_STEPS = 5
LEARNING_RATE = 0.002
BATCH_SIZE = 800
OPTIMIZER = "Adam"




#predicted_result = None
exp = None
predicted_prob = None
prediction_set = None
predicted_class = None


#Model Parameters
n1= 80   
n2=40
n3= 20

def model_fn(features, targets, mode, params):
  """Model function for Estimator."""
  
  # Comy_estimatorect the first hidden layer to input layer
  first_hidden_layer = tf.layers.dense(tf.layers.batch_normalization(tf.to_double(features)), n1, activation=tf.nn.relu)
  
  first_processed = tf.contrib.layers.dropout(first_hidden_layer, keep_prob=0.7)


  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(first_processed, n2, activation=tf.nn.relu)
  
  second_processed = tf.contrib.layers.dropout(second_hidden_layer,0.7)  
  
  third_hidden_layer = tf.layers.dense(second_processed, n3, activation=tf.nn.relu)
  
  third_processed = tf.contrib.layers.dropout(third_hidden_layer,0.7)

  # Comy_estimatorect the output layer to second hidden layer (no activation fn)
  logits = tf.layers.dense(third_processed, 2, activation=None)
  
  weights = tf.constant(params["weights"])
  #logits = tf.contrib.layers.layer_norm(pre_logits,activation_fn=None)
  
  
  # Generate Predictions
  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  # Calculate loss
  onehot_labels = tf.reshape(tf.contrib.layers.one_hot_encoding(targets, 2),[-1, 2])
  

  '''
  loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, weights=weights)
  loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
  '''

  #loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, weights=weights)
  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.TRAIN:
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    loss = tf.losses.softmax_cross_entropy(onehot_labels, logits, weights=weights)

    train_op = tf.contrib.layers.optimize_loss(
              loss=loss,
              global_step=tf.contrib.framework.get_global_step(),
              learning_rate=params["learning_rate"],
              optimizer= OPTIMIZER) 
      
  # Return a ModelFnOps object (eval_metrics not included)
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def input_fn(data_set):
  features = tf.constant(np.delete(data_set, 0, 1))
  labels = tf.constant(np.int_(np.delete(data_set, np.s_[1:], 1)))
  return features, labels

def new_input_fn(data_set):
  features = tf.constant(data_set)
  labels = tf.constant(np.int_(np.delete(data_set, np.s_[1:], 1)))
  return features, labels


def main():

  global prediction_set
  global training_weight
  global training_set
  '''
  all_set = pd.read_csv(DIR, skipinitialspace=True,
                             skiprows=0, usecols=COLUMNS).as_matrix()
  
  SORT = list(range(0,89))
  SORT.insert(0,89)
  all_set = all_set[:,np.array(SORT)]
  '''
  #np.random.shuffle(all_set)
  

  training_weight=training_set[:,-1]
  training_set=training_set[:,:-1]  

  
  model_params = {"learning_rate": LEARNING_RATE, "model_dir": MODEL_DIR, "weights": training_weight}
  configs = tf.contrib.learn.RunConfig(save_summary_steps=500)

#Build the estimator model
  my_estimator = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params, 
                                            config=configs, 
                                            model_dir= MODEL_DIR)

  
  
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=lambda: input_fn(training_set),
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

    
    
    
  #Initialize the training
  my_estimator.fit(input_fn=lambda: input_fn(training_set), steps=TRAINING_STEPS)
  
  
  
  
  #SKCompat Version (accepts using batch size)


  #Validate
  my_estimator.evaluate(input_fn=lambda: input_fn(validation_set))

  
  #Predict
  predicted_result = my_estimator.predict(input_fn=lambda: new_input_fn(prediction_set),as_iterable=False)
  predicted_prob = predicted_result["probabilities"]
  predicted_class = predicted_result["classes"]

  
  #Save 
  np.save('result.npy',predicted_prob)
  np.savetxt('result.csv',predicted_prob,delimiter=',')



if __name__ == "__main__":
    main()