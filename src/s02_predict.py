#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Mathias
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

import os
import sys
#import torch
import numpy as np
import pandas as pd 

#Imports the util module and network architectures for NetTCR
#If changes are wanted, change this directory to your own directory
#sys.path.append("/home/projects/vaccine/people/matjen/master_project/nettcr_src")

#Directory with the "keras_utils.py" script
#sys.path.append("/home/projects/vaccine/people/matjen/master_project/keras_src")
#Directory with the model architecure

import s99_project_functions
import random

# Set random seed
seed=15
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
tf.random.set_seed(seed) # Tensorflow random seed

### Input/Output ###
# Read in data
data = pd.read_csv('../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_final.csv')

# Directories
####################
##CHANGE PREFIX##
####################
experiment_index = '01'

#Define the list of peptides in the training data
pep_list = list(data[data.binder==1].peptide.value_counts(ascending=False).index)

### Model training parameters ###
train_parts = {0, 1, 2, 3, 4} #Partitions
dropout_rate = 0.6 #Dropout Rate
encoding = s99_project_functions.blosum50_20aa #Encoding for amino acid sequences

#Padding to certain length
a1_max = 7
a2_max = 8
a3_max = 22
b1_max = 6
b2_max = 7
b3_max = 23
pep_max = 12

# Define support functions for cuda tensors
#def get_variable(x):
#    """ Converts tensors to cuda, if available. """
#    if torch.cuda.is_available():
#        return x.cuda()
#    return x

#def get_numpy(x):
#    """ Get numpy array for both cuda and not. """
#    if torch.cuda.is_available():
#        return x.cpu().detach().numpy()
#    return x.data.numpy()

def my_numpy_function(y_true, y_pred):    
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        #Exception for when a positive observation is not present in a batch
        auc = np.array([float(0)])
    return auc

#Custom metric for AUC 0.1
def auc_01(y_true, y_pred):
    "Allows Tensorflow to use the function during training"
    auc_01 = tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)
    return auc_01

#Necessary to load the model with the custom metric
dependencies = {
    'auc_01': auc_01
}
     
def make_tf_ds(df, encoding):
    """Prepares the embedding for the input features to the model"""
    encoded_pep = s99_project_functions.enc_list_bl_max_len(df.peptide, encoding, pep_max)/5
    encoded_a1 = s99_project_functions.enc_list_bl_max_len(df.A1, encoding, a1_max)/5
    encoded_a2 = s99_project_functions.enc_list_bl_max_len(df.A2, encoding, a2_max)/5
    encoded_a3 = s99_project_functions.enc_list_bl_max_len(df.A3, encoding, a3_max)/5
    encoded_b1 = s99_project_functions.enc_list_bl_max_len(df.B1, encoding, b1_max)/5
    encoded_b2 = s99_project_functions.enc_list_bl_max_len(df.B2, encoding, b2_max)/5
    encoded_b3 = s99_project_functions.enc_list_bl_max_len(df.B3, encoding, b3_max)/5
    targets = df.binder.values
    tf_ds = [encoded_pep,
             encoded_a1, encoded_a2, encoded_a3, 
             encoded_b1, encoded_b2, encoded_b3,
             targets]

    return tf_ds
 
#Prepare output dataframe (test predictions)
pred_df = pd.DataFrame()


#Loop over each model
for t in train_parts:
    x_test_df = data[(data.partition==t)].reset_index()
    test_tensor = make_tf_ds(x_test_df, encoding = encoding)
    x_test = test_tensor[0:7]
    targets_test = test_tensor[7]
    avg_prediction = 0 #Reset prediction
    
    for v in train_parts:
        if v!=t:
            
            #Loading the model
            model = keras.models.load_model('../checkpoint/s01_e{}_t{}v{}'.format(experiment_index,t,v), custom_objects=dependencies)

            #Prediction by one model
            avg_prediction += model.predict(x = {"pep": x_test[0],
                                                 "a1": x_test[1],
                                                 "a2": x_test[2],
                                                 "a3": x_test[3],
                                                 "b1": x_test[4],
                                                 "b2": x_test[5],
                                                 "b3": x_test[6]})

            #Clears the session for the next model
            tf.keras.backend.clear_session()

    #Averaging the predictions between all models in the inner loop
    avg_prediction = avg_prediction/4
    x_test_df['prediction'] = avg_prediction
    pred_df = pd.concat([pred_df, x_test_df])

# Save predictions
pred_df.to_csv(outdir + '../data/s02_e{}_predictions.csv'.format(experiment_index,t,v), index=False)
