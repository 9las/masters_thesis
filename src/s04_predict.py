#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
import os
import sys
import numpy as np
import pandas as pd
import s99_project_functions
import argparse
import yaml

#Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
args = parser.parse_args()
config_filename = args.config

# Load config
config = s99_project_functions.load_config(config_filename)

# Set parameters from config
model_index = config['default']['model_index']
embedder_name_tcr = config['default']['embedder_name_tcr']
embedder_source_tcr = config['default']['embedder_source_tcr']
embedder_name_peptide = config['default']['embedder_name_peptide']
embedder_source_peptide = config['default']['embedder_source_peptide']
padding_value_peptide = config['default']['padding_value_peptide']
padding_side_peptide = config['default']['padding_side_peptide']
truncating_side_peptide = config['default']['truncating_side_peptide']
padding_value_tcr = config['default']['padding_value_tcr']
padding_side_tcr = config['default']['padding_side_tcr']
truncating_side_tcr = config['default']['truncating_side_tcr']
peptide_normalization_divisor = config['default']['peptide_normalization_divisor']
tcr_normalization_divisor = config['default']['tcr_normalization_divisor']
seed=config['default']['seed']
data_filename = config['default']['data_filename']
model_architecture_name = config['default']['model_architecture_name']

# Set random seed
keras.utils.set_random_seed(seed)

### Input/Output ###
# Read in data
data = pd.read_csv(filepath_or_buffer = os.path.join('../data/raw',
                                                     data_filename),
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

# Get dataframe with unique encoded peptides
if embedder_source_peptide == 'in-house':

    df_peptides = (s99_project_functions
                   .encode_unique_peptides(df = data,
                                           encoding_name = embedder_name_peptide))

else:
    df_peptides = pd.read_pickle(filepath_or_buffer = ('../data/s01_embedding_peptide_{}_{}.pkl'
                                                       .format(embedder_source_peptide,
                                                               embedder_name_peptide.replace('/', '_'))))

# Get dataframe with unique encoded CDRs
if embedder_source_tcr == 'in-house':

    df_tcrs = (s99_project_functions
               .encode_unique_tcrs(df = data,
                                   encoding_name = embedder_name_tcr))
else:
    df_tcrs = pd.read_pickle(filepath_or_buffer = ('../data/s01_embedding_tcr_{}_{}.pkl'
                                                   .format(embedder_source_tcr,
                                                           embedder_name_tcr.replace('/', '_'))))

# Pad unique peptides and CDRs
df_peptides = (s99_project_functions
               .pad_unique_peptides(df = df_peptides,
                                    padding_value = padding_value_peptide,
                                    padding_side = padding_side_peptide,
                                    truncating_side = truncating_side_peptide))
df_tcrs = (s99_project_functions
           .pad_unique_tcrs(df = df_tcrs,
                            padding_value = padding_value_tcr,
                            padding_side = padding_side_tcr,
                            truncating_side = truncating_side_tcr))

df_peptides = (df_peptides
               .drop(labels = 'count',
                     axis = 1))

# Join unique encoded sequences onto the data set
data = (data
        .merge(right = df_peptides,
               how = 'left',
               on = 'peptide')
        .merge(right = df_tcrs,
               how = 'left',
               on = 'original_index'))

# Remove unused variables from memory
del df_peptides, df_tcrs

# Do the prediction
partitions_count = 5
for t in range(partitions_count):

    # Get test data
    test_partition_mask = data['partition'] == t
    df_test = data[test_partition_mask]

    # Get model input
    peptide_test = np.stack(arrays = df_test['peptide_encoded'])
    a1_test = np.stack(arrays = df_test['a1_encoded'])
    a2_test = np.stack(arrays = df_test['a2_encoded'])
    a3_test = np.stack(arrays = df_test['a3_encoded'])
    b1_test = np.stack(arrays = df_test['b1_encoded'])
    b2_test = np.stack(arrays = df_test['b2_encoded'])
    b3_test = np.stack(arrays = df_test['b3_encoded'])

    if model_architecture_name == 'ff_CDR123':
        # Flatten peptide array
        peptide_test = np.reshape(a = peptide_test,
                                  newshape = (peptide_test.shape[0],
                                              -1))

        # Reduce embeddings of CDRs to be per CDR instead of per amino acid
        a1_test = np.mean(a = a1_test,
                          axis = 1)
        a2_test = np.mean(a = a2_test,
                          axis = 1)
        a3_test = np.mean(a = a3_test,
                          axis = 1)
        b1_test = np.mean(a = b1_test,
                          axis = 1)
        b2_test = np.mean(a = b2_test,
                          axis = 1)
        b3_test = np.mean(a = b3_test,
                          axis = 1)

    # Normalise embeddings
    peptide_test = peptide_test / peptide_normalization_divisor
    a1_test = a1_test / tcr_normalization_divisor
    a2_test = a2_test / tcr_normalization_divisor
    a3_test = a3_test / tcr_normalization_divisor
    b1_test = b1_test / tcr_normalization_divisor
    b2_test = b2_test / tcr_normalization_divisor
    b3_test = b3_test / tcr_normalization_divisor

    # Remove unused variables from memory
    del df_test

    # Initialize
    avg_prediction = 0

    for v in range(partitions_count):
        if v!=t:

            # Load the model
            model = keras.models.load_model(filepath = '../checkpoint/s02_m{}_t{}v{}'.format(model_index, t, v),
                                            custom_objects = {'auc01': s99_project_functions.auc01})

            # Do prediction by one model
            avg_prediction += model.predict(x = {'pep': peptide_test,
                                                 'a1': a1_test,
                                                 'a2': a2_test,
                                                 'a3': a3_test,
                                                 'b1': b1_test,
                                                 'b2': b2_test,
                                                 'b3': b3_test})

            # Clear the session for the next model
            tf.keras.backend.clear_session()

    # Average the predictions between all models in the inner loop
    avg_prediction = avg_prediction / 4
    data.loc[test_partition_mask, 'prediction'] = avg_prediction

# Select columns for output
data = (data.
        filter(items = ['A1',
                        'A2',
                        'A3',
                        'B1',
                        'B2',
                        'B3',
                        'peptide',
                        'binder',
                        'partition',
                        'original_index',
                        'prediction']))

# Save predictions
data.to_csv(path_or_buf = '../data/s04_m{}_predictions.tsv'.format(model_index),
            sep = '\t',
            index = False)
