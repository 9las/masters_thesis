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
import datasets

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
embedder_source_tcr = config['default']['embedder_source_tcr']
embedder_name_peptide = config['default']['embedder_name_peptide']
embedder_source_peptide = config['default']['embedder_source_peptide']
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
if embedder_source_peptide == 'in-house':

    df_peptides = (s99_project_functions
                   .encode_unique_peptides(df = data,
                                           encoding_name = embedder_name_peptide))

else:
    df_peptides = pd.read_pickle(filepath_or_buffer = ('../data/s01_embedding_peptide_{}_{}.pkl'
                                                       .format(embedder_source_peptide,
                                                               embedder_name_peptide.replace('/', '_'))))

# Functions
def map_embeddings(x):
    result_dict = dict()
    peptide = x['peptide']
    original_index = x['original_index']

    result_dict['peptide_encoded'] = df_peptides.loc[peptide, 'peptide_encoded']
    result_dict['weight'] = df_peptides.loc[peptide, 'weight']
    result_dict['a1_encoded'] = df_tcrs.loc[original_index, 'a1_encoded']
    result_dict['a2_encoded'] = df_tcrs.loc[original_index, 'a2_encoded']
    result_dict['a3_encoded'] = df_tcrs.loc[original_index, 'a3_encoded']
    result_dict['b1_encoded'] = df_tcrs.loc[original_index, 'b1_encoded']
    result_dict['b2_encoded'] = df_tcrs.loc[original_index, 'b2_encoded']
    result_dict['b3_encoded'] = df_tcrs.loc[original_index, 'b3_encoded']

    return result_dict

def extract_weights_from_dict(x):
    inputs = x[0]
    label = x[1]
    weight = inputs.pop('weight')

    return((inputs, label, weight))

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

# Normalise embeddings
df_tcrs /= tcr_normalization_divisor
df_peptides['peptide_encoded'] /= peptide_normalization_divisor

if model_architecture_name == 'ff_CDR123':
    df_peptides['peptide_encoded'] = (df_peptides['peptide_encoded']
                                      .map(arg = lambda x: x.flatten()))
    df_tcrs = (df_tcrs
               .applymap(func = lambda x: np.mean(a = x,
                                                  axis = 0)))
del data

features = datasets.Features({'original_index': datasets.Value('int32'),
                              'peptide': datasets.Value('string'),
                              'partition': datasets.Value('int8'),
                              'binder': datasets.Value('bool')})

data = datasets.load_dataset(path = 'csv',
                             data_files = '../data/raw/nettcr_train_swapped_peptide_ls_3_26_peptides_full_tcr_final.csv',
                             features = features)

if peptide_selection is not None:
# Select specific peptide to train model on
    data = (data
            .filter(lambda x: x['peptide'] == peptide_selection))

# Get training data
df_train = (data
            .filter(lambda x: (x['partition'] != t) & (x['partition'] != v))
            .map(map_embeddings))

# Get validation data
df_validation = (data
                 .filter(lambda x: x['partition'] == v)
                 .map(map_embeddings))

df_train = df_train['train'].to_tf_dataset(columns = ['peptide',
                                                      'a1_encoded',
                                                      'a2_encoded',
                                                      'a3_encoded',
                                                      'b1_encoded',
                                                      'b2_encoded',
                                                      'b3_encoded',
                                                      'weight'],
                                           label_cols = 'binder',
                                           batch_size = batch_size,
                                           shuffle = True)

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
