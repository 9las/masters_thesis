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
experiment_index = config['default']['experiment_index']
patience = config['default']['patience'] #Patience for Early Stopping
dropout_rate = config['default']['dropout_rate'] #Dropout Rate
epochs = config['default']['epochs'] #Number of epochs in the training
batch_size = config['default']['batch_size'] #Number of elements in each batch
embedding_name_tcr = config['default']['embedding_name_tcr']
embedding_name_peptide = config['default']['embedding_name_peptide']
padding_value_peptide = config['default']['padding_value_peptide']
padding_side_peptide = config['default']['padding_side_peptide']
truncating_side_peptide = config['default']['truncating_side_peptide']
padding_value_tcr = config['default']['padding_value_tcr']
padding_side_tcr = config['default']['padding_side_tcr']
truncating_side_tcr = config['default']['truncating_side_tcr']
weight_peptides = config['default']['weight_peptides']
seed = config['default']['seed']
data_file_name = config['default']['data']
model_name = config['default']['model']
peptide_selection = config['default']['peptide_selection']
peptide_normalization_divisor = config['default']['peptide_normalization_divisor']
tcr_normalization_divisor = config['default']['tcr_normalization_divisor']
learning_rate = config['default']['learning_rate']

# Set random seed
keras.utils.set_random_seed(seed)

# Set up keras to use mixed precision if on GPU
if tf.config.list_physical_devices('GPU'):
    keras.mixed_precision.set_global_policy('mixed_float16')
    mixed_precision = True
else:
    mixed_precision = False

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

# Get dataframe with unique encoded peptides
if embedding_name_peptide in {'blosum50_20aa',
                              'blosum50',
                              'one_hot',
                              'one_hot_20aa',
                              'amino_to_idx',
                              'phys_chem',
                              'blosum62',
                              'blosum62_20aa'}:

    df_peptides = (s99_project_functions
                   .encode_unique_peptides(df = data,
                                           encoding_name = embedding_name_peptide))

else:
    df_peptides = pd.read_pickle(filepath_or_buffer = ('../data/s01_embedding_peptide_{}.pkl'
                                                       .format(embedding_name_peptide)))

# Get dataframe with unique encoded CDRs
if embedding_name_tcr in {'blosum50_20aa',
                          'blosum50',
                          'one_hot',
                          'one_hot_20aa',
                          'amino_to_idx',
                          'phys_chem',
                          'blosum62',
                          'blosum62_20aa'}:

    df_tcrs = (s99_project_functions
               .encode_unique_tcrs(df = data,
                                   encoding_name = embedding_name_tcr))
else:
    df_tcrs = pd.read_pickle(filepath_or_buffer = ('../data/s01_embedding_tcr_{}.pkl'
                                                   .format(embedding_name_tcr)))

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
peptide_length = peptide_train.shape[1]
a1_length = a1_train.shape[1]
a2_length = a2_train.shape[1]
a3_length = a3_train.shape[1]
b1_length = b1_train.shape[1]
b2_length = b2_train.shape[1]
b3_length = b3_train.shape[1]
embedding_size_peptide = peptide_train.shape[2]
embedding_size_tcr = a1_train.shape[2]

model = getattr(s98_models, model_name)
model = model(dropout_rate = dropout_rate,
              seed = seed,
              embedding_size_peptide = embedding_size_peptide,
              embedding_size_tcr = embedding_size_tcr,
              a1_length = a1_length,
              a2_length = a2_length,
              a3_length = a3_length,
              b1_length = b1_length,
              b2_length = b2_length,
              b3_length = b3_length,
              peptide_length = peptide_length,
              mixed_precision = mixed_precision)

# Compile model
auc01 = s99_project_functions.auc01
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
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
history = model.fit(x = {'pep': peptide_train,
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
                                 keras.callbacks.ModelCheckpoint(filepath = '../checkpoint/s02_e{}_t{}v{}'.format(experiment_index,t,v),
                                                                 monitor = 'val_auc01',
                                                                 mode = 'max',
                                                                 save_best_only = True)])

# Get loss and metrics for each epoch during training
valid_loss = history.history['val_loss']
train_loss = history.history['loss']
valid_auc01 = history.history['val_auc01']

# Record metrics at checkpoint
valid_best = valid_loss[np.argmax(valid_auc01)+1]
best_epoch = np.argmax(valid_auc01)+1
best_auc01 = np.max(valid_auc01)

# Create directory to save results in
if not os.path.exists('../results'):
    os.makedirs('../results')

# Record metrics at checkpoint to file
outfile = open('../results/s02_e{}_validation_t{}v{}.tsv'.format(experiment_index,t,v), mode = 'w')
print('t', 'v', 'valid_loss', 'best_auc_0.1', 'best_epoch', sep = '\t', file = outfile)
print(t, v, valid_best, best_auc01, best_epoch, sep = '\t', file = outfile)
outfile.close()

# Prepare plotting
sns.set()
fig, ax = plt.subplots(figsize=(15, 10))

#Plotting the losses
ax.plot(train_loss, label='train')
ax.plot(valid_loss, label='validation')
ax.set_ylabel('Loss')
ax.set_xlabel('Epoch')
ax.legend()
ax.set_title('Model t.'+str(t)+'.v.'+str(v))

#Save training/validation loss plot
plt.tight_layout()
plt.show()
fig.savefig('../results/s02_e{}_learning_curves_t{}v{}.png'.format(experiment_index,t,v), dpi=200)
