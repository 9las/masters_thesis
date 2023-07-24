#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
import os
import sys
import numpy as np
import pandas as pd
import s99_project_functions
import s98_models
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

#Input arguments for running the model
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
#Test partition, which is excluded from training
parser.add_argument('-t', '--test_partition')
#Validation partition for early stopping
parser.add_argument('-v', '--validation_partition')
args = parser.parse_args()
config_filename = args.config
t = int(args.test_partition)
v = int(args.validation_partition)

# Load config
config = s99_project_functions.load_config(config_filename)

# Set parameters from config
model_index = config['default']['model_index']
patience = config['default']['patience'] #Patience for Early Stopping
dropout_rate = config['default']['dropout_rate'] #Dropout Rate
epochs = config['default']['epochs'] #Number of epochs in the training
batch_size = config['default']['batch_size'] #Number of elements in each batch
embedder_name_tcr = config['default']['embedder_name_tcr']
embedder_name_peptide = config['default']['embedder_name_peptide']
padding_value_peptide = config['default']['padding_value_peptide']
padding_side_peptide = config['default']['padding_side_peptide']
truncating_side_peptide = config['default']['truncating_side_peptide']
padding_value_tcr = config['default']['padding_value_tcr']
padding_side_tcr = config['default']['padding_side_tcr']
truncating_side_tcr = config['default']['truncating_side_tcr']
weight_peptides = config['default']['weight_peptides']
seed = config['default']['seed']
data_filename = config['default']['data_filename']
model_architecture_name = config['default']['model_architecture_name']
peptide_selection = config['default']['peptide_selection']
peptide_normalization_divisor = config['default']['peptide_normalization_divisor']
tcr_normalization_divisor = config['default']['tcr_normalization_divisor']
learning_rate = config['default']['learning_rate']
convolution_filters_count = config['default']['convolution_filters_count']
hidden_units_count = config['default']['hidden_units_count']

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
if embedder_name_peptide in {'blosum50_20aa',
                              'blosum50',
                              'one_hot',
                              'one_hot_20aa',
                              'amino_to_idx',
                              'phys_chem',
                              'blosum62',
                              'blosum62_20aa'}:

    df_peptides = (s99_project_functions
                   .encode_unique_peptides(df = data,
                                           encoding_name = embedder_name_peptide))

else:
    df_peptides = pd.read_pickle(filepath_or_buffer = ('../data/s01_embedding_peptide_{}.pkl'
                                                       .format(embedder_name_peptide)))

# Get dataframe with unique encoded CDRs
if embedder_name_tcr in {'blosum50_20aa',
                          'blosum50',
                          'one_hot',
                          'one_hot_20aa',
                          'amino_to_idx',
                          'phys_chem',
                          'blosum62',
                          'blosum62_20aa'}:

    df_tcrs = (s99_project_functions
               .encode_unique_tcrs(df = data,
                                   encoding_name = embedder_name_tcr))
else:
    df_tcrs = pd.read_pickle(filepath_or_buffer = ('../data/s01_embedding_tcr_{}.pkl'
                                                   .format(embedder_name_tcr)))

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

# Weight peptides
if weight_peptides:
    peptide_count = data.shape[0]
    peptide_unique_count = df_peptides.shape[0]

    df_peptides = (df_peptides
    .assign(weight = lambda x: (np.log2(peptide_count
                                        / x['count'])
                                / np.log2(peptide_unique_count)))
    .assign(weight = lambda x: (x['weight']
                                * (peptide_count
                                   / np.sum(x['weight']
                                            * x['count'])))))
else:
    df_peptides = (df_peptides
                   .assign(weight = 1))

df_peptides = (df_peptides
               .drop(labels = 'count',
                     axis = 1))

if peptide_selection is not None:
# Select specific peptide to train model on
    data = (data
            .query('peptide == @peptide_selection'))

# Remove test partition and join unique encoded sequences onto the data set
data = (data
        .query('partition != @t')
        .merge(right = df_peptides,
               how = 'left',
               on = 'peptide')
        .merge(right = df_tcrs,
               how = 'left',
               on = 'original_index')
        .filter(items = ['peptide_encoded',
                         'a1_encoded',
                         'a2_encoded',
                         'a3_encoded',
                         'b1_encoded',
                         'b2_encoded',
                         'b3_encoded',
                         'weight',
                         'binder',
                         'partition']))

# Get training data
df_train = (data
            .query('partition != @v'))

# Get validation data
df_validation = (data
                 .query('partition == @v'))

# Get model input
peptide_train = np.stack(arrays = df_train['peptide_encoded'])
a1_train = np.stack(arrays = df_train['a1_encoded'])
a2_train = np.stack(arrays = df_train['a2_encoded'])
a3_train = np.stack(arrays = df_train['a3_encoded'])
b1_train = np.stack(arrays = df_train['b1_encoded'])
b2_train = np.stack(arrays = df_train['b2_encoded'])
b3_train = np.stack(arrays = df_train['b3_encoded'])
binder_train = np.asarray(df_train['binder'])
weight_train = np.asarray(df_train['weight'])

peptide_validation = np.stack(arrays = df_validation['peptide_encoded'])
a1_validation = np.stack(arrays = df_validation['a1_encoded'])
a2_validation = np.stack(arrays = df_validation['a2_encoded'])
a3_validation = np.stack(arrays = df_validation['a3_encoded'])
b1_validation = np.stack(arrays = df_validation['b1_encoded'])
b2_validation = np.stack(arrays = df_validation['b2_encoded'])
b3_validation = np.stack(arrays = df_validation['b3_encoded'])
binder_validation = np.asarray(df_validation['binder'])
weight_validation = np.asarray(df_validation['weight'])

if model_architecture_name == 'ff_CDR123':
    # Flatten peptide array
    peptide_train = np.reshape(a = peptide_train,
                               newshape = (peptide_train.shape[0],
                                           -1))
    peptide_validation = np.reshape(a = peptide_validation,
                               newshape = (peptide_validation.shape[0],
                                           -1))

    # Reduce embeddings of CDRs to be per CDR instead of per amino acid
    a1_train = np.mean(a = a1_train,
                       axis = 1)
    a2_train = np.mean(a = a2_train,
                       axis = 1)
    a3_train = np.mean(a = a3_train,
                       axis = 1)
    b1_train = np.mean(a = b1_train,
                       axis = 1)
    b2_train = np.mean(a = b2_train,
                       axis = 1)
    b3_train = np.mean(a = b3_train,
                       axis = 1)


    a1_validation = np.mean(a = a1_validation,
                            axis = 1)
    a2_validation = np.mean(a = a2_validation,
                            axis = 1)
    a3_validation = np.mean(a = a3_validation,
                            axis = 1)
    b1_validation = np.mean(a = b1_validation,
                            axis = 1)
    b2_validation = np.mean(a = b2_validation,
                            axis = 1)
    b3_validation = np.mean(a = b3_validation,
                            axis = 1)

# Normalise embeddings
peptide_train = peptide_train / peptide_normalization_divisor
a1_train = a1_train / tcr_normalization_divisor
a2_train = a2_train / tcr_normalization_divisor
a3_train = a3_train / tcr_normalization_divisor
b1_train = b1_train / tcr_normalization_divisor
b2_train = b2_train / tcr_normalization_divisor
b3_train = b3_train / tcr_normalization_divisor

peptide_validation = peptide_validation / peptide_normalization_divisor
a1_validation = a1_validation / tcr_normalization_divisor
a2_validation = a2_validation / tcr_normalization_divisor
a3_validation = a3_validation / tcr_normalization_divisor
b1_validation = b1_validation / tcr_normalization_divisor
b2_validation = b2_validation / tcr_normalization_divisor
b3_validation = b3_validation / tcr_normalization_divisor

# Get model architecture
model_architecture = getattr(s98_models, model_architecture_name)
peptide_shape = peptide_train.shape[1:]

if model_architecture_name == 'CNN_CDR123_global_max':
    a1_shape = a1_train.shape[1:]
    a2_shape = a2_train.shape[1:]
    a3_shape = a3_train.shape[1:]
    b1_shape = b1_train.shape[1:]
    b2_shape = b2_train.shape[1:]
    b3_shape = b3_train.shape[1:]

    model_architecture = model_architecture(dropout_rate = dropout_rate,
                                            seed = seed,
                                            peptide_shape = peptide_shape,
                                            a1_shape = a1_shape,
                                            a2_shape = a2_shape,
                                            a3_shape = a3_shape,
                                            b1_shape = b1_shape,
                                            b2_shape = b2_shape,
                                            b3_shape = b3_shape,
                                            convolution_filters_count = convolution_filters_count,
                                            hidden_units_count = hidden_units_count)

elif model_architecture_name == 'ff_CDR123':
    cdr_shape = a1_train.shape[1:]

    model_architecture = model_architecture(dropout_rate = dropout_rate,
                                            seed = seed,
                                            peptide_shape = peptide_shape,
                                            cdr_shape = cdr_shape,
                                            hidden_units_count = hidden_units_count)

# Compile model
auc01 = s99_project_functions.auc01
model_architecture.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                           loss = tf.keras.losses.BinaryCrossentropy(),
                           metrics = [auc01, 'AUC'],
                           weighted_metrics = [])

# Remove unused variables from memory
del df_train, df_validation, df_peptides, df_tcrs, data

#Creates the directory to save the model in
if not os.path.exists('../checkpoint'):
    os.makedirs('../checkpoint')

# Announce training
print('Training model with test_partition = {} & validation_partition = {}'.format(t,v), end = '\n')

# Train model
history = model_architecture.fit(x = {'pep': peptide_train,
                                      'a1': a1_train,
                                      'a2': a2_train,
                                      'a3': a3_train,
                                      'b1': b1_train,
                                      'b2': b2_train,
                                      'b3': b3_train},
                                 y = binder_train,
                                 batch_size = batch_size,
                                 epochs = epochs,
                                 verbose = 2,
                                 sample_weight = weight_train,
                                 validation_data = ({'pep': peptide_validation,
                                                     'a1': a1_validation,
                                                     'a2': a2_validation,
                                                     'a3': a3_validation,
                                                     'b1': b1_validation,
                                                     'b2': b2_validation,
                                                     'b3': b3_validation},
                                                    binder_validation,
                                                    weight_validation),
                                 validation_batch_size = batch_size,
                                 shuffle = True,
                                 callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_auc01',
                                                                            mode = 'max',
                                                                            patience = patience),
                                              keras.callbacks.ModelCheckpoint(filepath = '../checkpoint/s02_m{}_t{}v{}'.format(model_index,t,v),
                                                                              monitor = 'val_auc01',
                                                                              mode = 'max',
                                                                              save_best_only = True)])

# Create directory to save results in
if not os.path.exists('../results'):
    os.makedirs('../results')

# Get training history
history_df = pd.DataFrame(data = history.history)
history_df.index.name = 'epoch'
history_df.index += 1

# Extract training history at maximum AUC 0.1
epoch_max_auc01 = history_df['val_auc01'].idxmax()
history_max_auc01 = history_df.loc[[epoch_max_auc01]]


# Write training history to files
history_df.to_csv(path_or_buf = '../results/s02_m{}_training_history_t{}v{}.tsv'.format(model_index,
                                                                                        t,
                                                                                        v),
                  sep = '\t')

history_max_auc01.to_csv(path_or_buf = '../results/s02_m{}_training_history_max_auc01_t{}v{}.tsv'.format(model_index,
                                                                                                         t,
                                                                                                         v),
                         sep = '\t')
