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
embedding_model_name = config['default']['embedding_model_name'] 
seed=config['default']['seed']
data_file_name = config['default']['data']


if embedding_model_name is None:
    use_embeddings = False
else:
    use_embeddings = True

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
partitions_count = 5
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

#Prepare output dataframe (test predictions)
pred_df = pd.DataFrame()


#Loop over each model
for t in range(partitions_count):
    x_test_df = data.query('partition == @t')
    test_tensor = (s99_project_functions
                   .get_model_input(df = x_test_df,
                                    df_peptides_unique = df_peptides_unique,
                                    peptides_unique_encoded = peptides_unique_encoded,
                                    use_embeddings = use_embeddings,
                                    encoding_scheme_name = encoding_scheme_name,
                                    a1_max = a1_max,
                                    a2_max = a2_max,
                                    a3_max = a3_max,
                                    b1_max = b1_max,
                                    b2_max = b2_max,
                                    b3_max = b3_max,
                                    get_weights = False))
    avg_prediction = 0 # Intilialize

    for v in range(partitions_count):
        if v!=t:

            #Loading the model
            model = keras.models.load_model(filepath = '../checkpoint/s01_e{}_t{}v{}'.format(experiment_index,t,v),
                                            custom_objects = {'auc_01': auc_01})

            #Prediction by one model
            avg_prediction += model.predict(x = {"pep": test_tensor['peptide'],
                                                 "a1": test_tensor['a1'],
                                                 "a2": test_tensor['a2'],
                                                 "a3": test_tensor['a3'],
                                                 "b1": test_tensor['b1'],
                                                 "b2": test_tensor['b2'],
                                                 "b3": test_tensor['b3']})

            #Clears the session for the next model
            tf.keras.backend.clear_session()

    #Averaging the predictions between all models in the inner loop
    avg_prediction = avg_prediction/4
    x_test_df['prediction'] = avg_prediction
    pred_df = pd.concat([pred_df, x_test_df])

# Save predictions
pred_df.to_csv('../data/s02_e{}_predictions.csv'.format(experiment_index), index=False)
