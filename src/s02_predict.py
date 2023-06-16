#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
import os
import sys
import numpy as np
import pandas as pd 
import s99_project_functions
import random
import argparse
import yaml

#Input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
args = parser.parse_args() 
config_filename = args.config

# Load config
config = s99_project_functions.load_config(config_filename)

# Set parameters from config
experiment_index = config['default']['experiment_index']
encoding_scheme_name = config['default']['encoding']
seed=config['default']['seed']
data_file_name = config['default']['data']

# Set random seed
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
tf.random.set_seed(seed) # Tensorflow random seed

### Input/Output ###
# Read in data
data = pd.read_csv(filepath_or_buffer = os.path.join('../data/raw',
                                                     data_file_name),
                   usecols = ['A1',
                              'A2',
                              'A3',
                              'B1',
                              'B2',
                              'B3',
                              'peptide',
                              'binder',
                              'partition',
                              'original_index'])

### Model training parameters ###
train_parts = {0, 1, 2, 3, 4} #Partitions
#encoding = getattr(s99_project_functions, config['default']['encoding']) #Encoding for amino acid sequences

# Make dataframe with unique peptides and their row number
df_peptides_unique = pd.DataFrame(index = pd.unique(data['peptide']))
peptides_unique_count = df_peptides_unique.shape[0]
df_peptides_unique = (df_peptides_unique
                      .assign(row_number = range(peptides_unique_count)))

# Encode unique peptides
peptides_unique_encoded = (df_peptides_unique
                           .index
                           .map(lambda x: (s99_project_functions
                                           .encode_peptide(sequence = x,
                                                           encoding_scheme_name = encoding_scheme_name)))
                           .tolist())

# Pad unique peptides
peptides_unique_encoded = s99_project_functions.pad_sequences(sequence_array = peptides_unique_encoded,
                                                              padding_value = -5)/5

# Get max sequence lengths for padding
a1_max = data['A1'].map(len).max()
a2_max = data['A2'].map(len).max()
a3_max = data['A3'].map(len).max()
b1_max = data['B1'].map(len).max()
b2_max = data['B2'].map(len).max()
b3_max = data['B3'].map(len).max()

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
pred_df.to_csv('../data/s02_e{}_predictions.csv'.format(experiment_index), index=False)
