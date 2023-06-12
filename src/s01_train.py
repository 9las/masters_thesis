#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:58:33 2023

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
#Directory with the "keras_utils.py" script
#sys.path.append("/home/projects/vaccine/people/matjen/master_project/keras_src")
#Directory with the model architecure
#sys.path.append("/home/projects/vaccine/people/matjen/master_project/keras_src")

import s99_project_functions
import s98_models
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse

#Input arguments for running the model
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config")
#Test partition, which is excluded from training
parser.add_argument("-t", "--test_partition")
#Validation partition for early stopping
parser.add_argument("-v", "--validation_partition")
args = parser.parse_args()
config_filename = args.config
t = int(args.test_partition)
v = int(args.validation_partition)

# Load config
config = s99_project_functions.load_config(config_filename)

# Set experiment index
experiment_index = config['default']['experiment_index']

#Import model
model = getattr(s98_models, config['default']['model'])

#Makes the plots look better
sns.set()

# Set random seed
seed = config['default']['seed']
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
tf.random.set_seed(seed) # Tensorflow random seed

### Input/Output ###
# Read in data
data = pd.read_csv(filepath_or_buffer = os.path.join('../data/raw', config['default']['data']),
                   usecols = ['peptide',
                              'binder',
                              'partition',
                              'original_index'])

peptides_unique = data['peptide'].unique()
peptides_unique_count = peptides_unique.size

#Sample weights

if config['default']['sample_weight']:
    weight_dict = np.log2(data.shape[0]/(data.peptide.value_counts()))/np.log2(peptides_unique_count)
#Normalize, so that loss is comparable
    weight_dict = weight_dict*(data.shape[0]/np.sum(weight_dict*data.peptide.value_counts()))
    data["sample_weight"] = data["peptide"].map(weight_dict)
else:
    data["sample_weight"] = 1

### Model training parameters ###
patience = config['default']['patience'] #Patience for Early Stopping
dropout_rate = config['default']['dropout_rate'] #Dropout Rate
encoding = getattr(s99_project_functions, config['default']['encoding']) #Encoding for amino acid sequences
EPOCHS = config['default']['epochs'] #Number of epochs in the training
batch_size = config['default']['batch_size'] #Number of elements in each batch

#Padding to certain length
a1_max = 7
a2_max = 8
a3_max = 22
b1_max = 6
b2_max = 7
b3_max = 23
pep_max = 12

def make_tf_ds(df, encoding):
    """Prepares the embedding for the input features to the model"""
    encoded_pep = s99_project_functions.enc_list_bl_max_len(df.peptide, encoding, pep_max)/5 # Divide by 5 to make numbers closer to the interval -1 to 1 - makes model learn better
    encoded_a1 = s99_project_functions.enc_list_bl_max_len(df.A1, encoding, a1_max)/5
    encoded_a2 = s99_project_functions.enc_list_bl_max_len(df.A2, encoding, a2_max)/5
    encoded_a3 = s99_project_functions.enc_list_bl_max_len(df.A3, encoding, a3_max)/5
    encoded_b1 = s99_project_functions.enc_list_bl_max_len(df.B1, encoding, b1_max)/5
    encoded_b2 = s99_project_functions.enc_list_bl_max_len(df.B2, encoding, b2_max)/5
    encoded_b3 = s99_project_functions.enc_list_bl_max_len(df.B3, encoding, b3_max)/5
    targets = df.binder.values
    sample_weights = df.sample_weight
    tf_ds = [encoded_pep,
             encoded_a1, encoded_a2, encoded_a3, 
             encoded_b1, encoded_b2, encoded_b3,
             targets,
             sample_weights]
    return tf_ds

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

#Creates the directory to save the model in
if not os.path.exists('../results'):
    os.makedirs('../results')
    
outfile = open("../results/s01_e{}_validation_t{}v{}.tsv".format(experiment_index,t,v), mode = "w")
print("t", "v", "valid_loss", "best_auc_0.1", "best_epoch", sep = "\t", file = outfile)


# Prepare plotting
fig, ax = plt.subplots(figsize=(15, 10))

#Training data
x_train_df = data[(data.partition!=t)&(data.partition!=v)].reset_index()
train_tensor = make_tf_ds(x_train_df, encoding = encoding)
x_train = train_tensor[0:7] #Inputs
targets_train = train_tensor[7] #Target (0 or 1)
weights_train = train_tensor[8] #Sample weight for the loss function

#Validation data - Used for early stopping
x_valid_df = data[(data.partition==v)]
valid_tensor = make_tf_ds(x_valid_df, encoding = encoding)
x_valid = valid_tensor[0:7] #Inputs
targets_valid = valid_tensor[7] #Target (0 or 1)
weights_valid = valid_tensor[8] #Sample weight for the loss function

#Selection of the model to train
model = model(dropout_rate = dropout_rate, seed = seed)

#Creates the directory to save the model in
if not os.path.exists('../checkpoint'):
    os.makedirs('../checkpoint')

#Saves the model at the best epoch (based on validation loss or other metric)
ModelCheckpoint = keras.callbacks.ModelCheckpoint(
        filepath = '../checkpoint/s01_e{}_t{}v{}'.format(experiment_index,t,v),
        monitor = "val_auc_01",
        mode = "max",
        save_best_only = True)

#EarlyStopping function used for stopping model training when the model does not improve
EarlyStopping = keras.callbacks.EarlyStopping(
    monitor = "val_auc_01",
    mode = "max",
    patience = patience)

#Callbacks to include for the model training
callbacks_list = [EarlyStopping,
                  ModelCheckpoint
    ]

#Optimizers, loss functions, and additional metrics to track
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [auc_01, "AUC"],
              weighted_metrics = [])

#Announce Training
print("Training model with test_partition = {} & validation_partition = {}".format(t,v), end = "\n")

#Model training
history = model.fit(x = {"pep": x_train[0],
                         "a1": x_train[1],
                         "a2": x_train[2],
                         "a3": x_train[3],
                         "b1": x_train[4],
                         "b2": x_train[5],
                         "b3": x_train[6]},
          y = targets_train,
          batch_size = batch_size,
          epochs = EPOCHS,
          verbose = 2,
          sample_weight = weights_train,
          validation_data = ({"pep": x_valid[0],
                              "a1": x_valid[1],
                              "a2": x_valid[2],
                              "a3": x_valid[3],
                              "b1": x_valid[4],
                              "b2": x_valid[5],
                              "b3": x_valid[6]}, 
                             targets_valid,
                             weights_valid),
          validation_batch_size = batch_size,
          shuffle = True,
          callbacks=callbacks_list
          )            

#Loss and metrics for each epoch during training            
valid_loss = history.history["val_loss"]
train_loss = history.history["loss"]
valid_auc = history.history["val_auc"]

#Saving the model at the last epoch of training
#model.save(outdir+'/checkpoint/'+'t.'+str(t)+'.v.'+str(v)+".h5")

#Plotting the losses
ax.plot(train_loss, label='train')
ax.plot(valid_loss, label='validation')
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.legend()
ax.set_title('Model t.'+str(t)+'.v.'+str(v))

#Record metrics at checkpoint
valid_best = valid_loss[np.argmax(valid_auc)+1]
best_epoch = np.argmax(valid_auc)+1
best_auc = np.max(valid_auc)

#Records loss and metrics at saved epoch
print(t, v, valid_best, best_auc, best_epoch, sep = "\t", file = outfile)

#Clears the session for the next model
tf.keras.backend.clear_session()

#Close log file
outfile.close()

#Save training/validation loss plot
plt.tight_layout()
plt.show()
fig.savefig('../results/s01_e{}_learning_curves_t{}v{}.png'.format(experiment_index,t,v), dpi=200)
