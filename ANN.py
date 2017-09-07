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

tf.logging.set_verbosity(tf.logging.INFO)


#Generate Training numbers and training weight
DIR = "../data/stock_train_data_20170901.csv"
COLUMNS = list(range(1,91))  #Read Feature,weight,label
all_set = pd.read_csv(DIR, skipinitialspace=True,
                             skiprows=0, usecols=COLUMNS).as_matrix()

SORT = list(range(0,89))
SORT.insert(0,89)   #89,0-87,88
all_set = all_set[:,np.array(SORT)] #Change into 0Label,Feature,89Weight
#np.random.shuffle(all_set)
training_set=all_set
training_weight=training_set[:,-1]  #last column
training_stat=training_set[:,:-1]   #Except last column:Label+Features	             
	



#Generate Testing numbers and training weight	
TESTDIR="../data/stock_test_data_20170901.csv"

SSD=list(range(1,89))   #1-88,Features
prediction_set=pd.read_csv(TESTDIR, skipinitialspace=True,
                             skiprows=0, usecols=SSD).as_matrix()

	

	
#Training Parameters
MODEL_DIR = "../data/model1"
TRAINING_STEPS =5
LEARNING_RATE = 0.002
BATCH_SIZE = 800
OPTIMIZER = "Adam"
model_params = {"learning_rate": LEARNING_RATE, "model_dir": MODEL_DIR, "weights": training_weight}




#predicted_result = None
exp = None
predicted_prob = None
prediction_set = None
predicted_class = None





#Model Parameters
n1= 88   
n2=44

n3= 2
n4= 22
n5= 2




def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

  
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

  # Comy_estimatorect the output layer to second hidden layer (no activation fn)
  logits = tf.layers.dense(third_processed, 2, activation=None)
  
  weights = tf.constant(params["weights"])
  #logits = tf.contrib.layers.layer_norm(pre_logits,activation_fn=None)
  
  
  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Calculate loss
  onehot_labels = tf.reshape(tf.contrib.layers.one_hot_encoding(targets, 2),[-1, 2])
  

  loss = tf.losses.softmax_cross_entropy(onehot_labels, logits) #, weights=weights
  

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

#Load datasets before
  configs = tf.contrib.learn.RunConfig(save_summary_steps=500)

#Build the estimator model
  my_estimator = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params, 
                                            config=configs, 
                                            model_dir= MODEL_DIR)

  
  
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=lambda: input_fn(training_stat),
    early_stopping_metric="loss",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

    
    
    
  #Initialize the training
  my_estimator.fit(input_fn=lambda: input_fn(training_stat), steps=TRAINING_STEPS)
  
  
  
  
  #SKCompat Version (accepts using batch size)
  global predicted_result
  global exp
  global predicted_prob
  
  #Removed the outside "list"
  predicted_result = my_estimator.predict(input_fn=lambda: new_input_fn(prediction_set),as_iterable=False)
  predicted_prob = predicted_result["probabilities"]
  predicted_class = predicted_result["classes"]

  
  #Save 
  np.save('result.npy',predicted_prob)
  np.savetxt('result.csv',predicted_prob,delimiter=',')



if __name__ == "__main__":
    main()